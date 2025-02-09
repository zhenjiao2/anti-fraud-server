import requests
import random
import json
from flask import (Flask, request, jsonify)
from dotenv import load_dotenv
from waitress import serve
import os
import ssl
import ai

from conversation import Conversation

app = Flask(__name__)

# Define valid platforms for OptPlatform
VALID_PLATFORMS = {"RESTAPI", "Web", "Android", "iOS", "Windows", "Mac", "iPad", "Unknown"}
suspect_level = 0
notice = ""
conversations = {}

@app.route('/health', methods=['GET'])
def health():
    return "OK", 200

@app.route('/callback', methods=['POST'])
def callback():
    load_dotenv()
    global suspect_level 
    global notice 
    global conversations

    # Get query parameters from the URL
    sdk_appid = request.args.get("SdkAppid")
    callback_command_param = request.args.get("CallbackCommand")
    opt_platform = request.args.get("OptPlatform")

    # ClientIP from query parameter is not needed since we can get it from request.remote_addr
    client_ip = request.remote_addr

    # Check that required query parameters are present and valid
    if not sdk_appid:
        return jsonify({"ActionStatus": "FAIL", "ErrorInfo": "SdkAppid missing", "ErrorCode": -1}), 400
    if sdk_appid != os.environ.get("SDK_APP_ID"):
        return jsonify({"ActionStatus": "FAIL", "ErrorInfo": "Invalid SdkAppid", "ErrorCode": -1}), 400
    
    if opt_platform not in VALID_PLATFORMS:
        return jsonify({"ActionStatus": "FAIL", "ErrorInfo": "Invalid OptPlatform", "ErrorCode": -1}), 400

    # Read JSON content from the POST body; expected to be the message content
    data = request.get_json()
    if not data:
        return jsonify({"ActionStatus": "FAIL", "ErrorInfo": "Missing or invalid JSON content", "ErrorCode": -1}), 400

    # After calling request.get_json(), you can retrieve the original message.
    original_message = data.get("MsgBody")  # gets the "MsgBody" from the JSON payload
    # Parse the message text from the original message.
    # Assuming original_message is a list of message elements.
    message_text = None
    if isinstance(original_message, list):
        for elem in original_message:
            if elem.get("MsgType") == "TIMTextElem":
                # Get the "Text" content from "MsgContent"
                message_text = elem.get("MsgContent", {}).get("Text")
                break
    
    response_json = {
        "ActionStatus": "OK",
        "ErrorInfo": "",
        "ErrorCode": 0,  # 0 means allow sending; you can change this value based on your logic.
    }
    from_account = data.get("From_Account")
    to_account = data.get("To_Account")
    key = min(from_account, to_account) + '-' + max(from_account, to_account)

    if callback_command_param == "C2C.CallbackAfterSendMsg":
        add_message(from_account, to_account, message_text)
        
        # call AI to check if it is related to fraud
        msg = conversations[key].get_lastmessages()
        print(f"[Request to AI]{msg}")
        ai_response = ai.check_suspect_level(msg)
        print(f"[AI Response] {ai_response}")

        def remove_codeblock_markers(text):
            # Remove the starting marker "```json"
            if text.startswith("```json"):
                text = text[len("```json"):].lstrip()
            # Remove the ending marker "```"
            if text.endswith("```"):
                text = text[:-3].rstrip()
            return text        
        
        data = json.loads(remove_codeblock_markers(ai_response))
        if data:
            suspect_level = data.get("suspect_level")
            if suspect_level and suspect_level >= 3:
                notice = f"---\n注意：{data.get("reasoning")}\n可疑程度：{suspect_level}\n建议：{data.get("advice")}\n---"

                if suspect_level >= 4:
                    url = "https://console.tim.qq.com/v4/openim/sendmsg"
                    params = {
                    "SdkAppid": sdk_appid,  
                    "identifier": os.environ.get("ADMIN_USER_ID"),
                    "usersig": os.environ.get("ADMIN_USER_SIG"),
                    "random": random.randint(1, 10000000),
                    "contenttype": "json"
                    }
                    payload = {
                        "SyncOtherMachine": 2, 
                        "To_Account": to_account,
                        "MsgRandom": random.randint(1, 10000000),
                        "ForbidCallbackControl":[
                            "ForbidBeforeSendMsgCallback",
                            "ForbidAfterSendMsgCallback"], 
                        "MsgBody": [
                            {
                                "MsgType": "TIMTextElem",
                                "MsgContent": {
                                    "Text": f"管理员提醒：你在和{from_account}的聊天中对方发送了可疑消息。\n{notice}"
                                }
                            }
                        ]
                    }                
                    print(f"Sending admin message to {to_account}:", payload)
                    response = requests.post(url, params=params, data=json.dumps(payload))
                    print("Status Code of admin message:", response)


    elif callback_command_param == "C2C.CallbackBeforeSendMsg":
        if suspect_level >= 3:
            print(f"\nReceived message from {from_account} to {to_account}:", message_text)
            response_json = {
                "ActionStatus": "OK",
                "ErrorInfo": "",
                "ErrorCode": 0,  # 0 means allow sending; you can change this value based on your logic.
                "MsgBody": [
                    # Using original message element(s) from request
                    {
                        "MsgType": "TIMTextElem",
                        "MsgContent": {
                            "Text": f"{notice}\n{message_text}"                        }
                    }
                    # Appending an additional custom message element
                ],
            }

            suspect_level = 0
            notice = ""
            
    return jsonify(response_json), 200

def add_message(from_account, to_account, message=None):
    key = min(from_account, to_account) + '-' + max(from_account, to_account)
    if key not in conversations:
        conversation = Conversation(from_account, to_account, message)
        conversations[key] = conversation
    else:
        conversation = conversations[key]
        conversation.add_message(from_account, message)

if __name__ == '__main__':
    app.run(debug=True)