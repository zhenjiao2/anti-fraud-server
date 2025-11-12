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

# 聊天服务器支持的平台，表示当前消息是从哪个平台发送而来，和反诈无关
VALID_PLATFORMS = {"RESTAPI", "Web", "Android", "iOS", "Windows", "Mac", "iPad", "Unknown"}
suspect_level = 0
notice = ""
# 记录所有聊天记录的dictionary类型的全局变量，保存的内容类似如下：
# {
#    "用户1-用户2": ["用户1: 你好", "用户2: 你好", "用户1: 你是谁", "用户2: 我是AI助手"],
#    "用户3-用户4": ["用户3: 你好", "用户4: 你好", "用户3: 你是谁", "用户4: 我是AI助手"]
# }
# 在代码中，可以通过conversations[key]来获取到对应的聊天记录，然后调用get_lastmessages()来获取最近的聊天记录
# 具体的数据结构定义在conversation.py中
conversations = {}

@app.route('/health', methods=['GET'])
def health():
    return "OK", 200

# 反诈服务器回调API入口，比如https://anti-fraud.com/callback
# 具体的文档可见https://cloud.tencent.com/document/product/269/1632?from=console_document_search
@app.route('/callback', methods=['POST'])
def callback():
    load_dotenv()
    #三个全局变量，suspect_level可疑程度, notice给用户的提示消息, conversations聊天记录
    global suspect_level 
    global notice 
    global conversations

    # 获取query参数,就是url中问号后面的参数，比如https://anti-fraud.com/callback?SdkAppid=123456&CallbackCommand=C2C.CallbackBeforeSendMsg&OptPlatform=RESTAPI
    # 这里的SdkAppid, CallbackCommand, OptPlatform是必须的参数，用来验证请求的合法性
    sdk_appid = request.args.get("SdkAppid")
    callback_command_param = request.args.get("CallbackCommand")
    opt_platform = request.args.get("OptPlatform")

    client_ip = request.remote_addr

    # 检查请求的合法性，如果缺少SdkAppid，或者SdkAppid不正确，返回错误信息
    if not sdk_appid:
        return jsonify({"ActionStatus": "FAIL", "ErrorInfo": "SdkAppid missing", "ErrorCode": -1}), 400
    if sdk_appid != os.environ.get("SDK_APP_ID"):
        return jsonify({"ActionStatus": "FAIL", "ErrorInfo": "Invalid SdkAppid", "ErrorCode": -1}), 400
    
    if opt_platform not in VALID_PLATFORMS:
        return jsonify({"ActionStatus": "FAIL", "ErrorInfo": "Invalid OptPlatform", "ErrorCode": -1}), 400

    # request是callback的输入参数，输入的内容是类似下面的json格式
    #     {
    #     "CallbackCommand": "C2C.CallbackBeforeSendMsg", // 回调命令
    #     "From_Account": "jared", // 发送者
    #     "To_Account": "Jonh", // 接收者
    #     "MsgSeq": 48374, // 消息序列号
    #     "MsgRandom": 2837546, // 消息随机数
    #     "MsgTime": 1557481126, // 消息的发送时间戳，单位为秒 
    #     "MsgKey": "48374_2837546_1557481126", //消息的唯一标识，可用于 REST API 撤回单聊消息
    #     "MsgId": "144115233406643804-1727580296-4026038328", // 消息在客户端上的唯一标识
    #     "OnlineOnlyFlag":1, //在线消息，为1，否则为0；
    #     "MsgBody": [ // 消息体，参见 TIMMessage 消息对象
    #         {
    #             "MsgType": "TIMTextElem", // 文本
    #             "MsgContent": {
    #                 "Text": "red packet"
    #             }
    #         }
    #     ],
    #     "CloudCustomData": "your cloud custom data",
    #     "EventTime": 1670574414123 //毫秒级别，事件触发时间戳
    # }
    data = request.get_json()
    if not data:
        return jsonify({"ActionStatus": "FAIL", "ErrorInfo": "Missing or invalid JSON content", "ErrorCode": -1}), 400

    # 根据上面的文件格式，MsgBody里面有消息的具体内容
    original_message = data.get("MsgBody")  
    # 用message_text来保存消息的字符串内容
    message_text = None
    if isinstance(original_message, list):
        for elem in original_message:
            if elem.get("MsgType") == "TIMTextElem":
                # Get the "Text" content from "MsgContent"
                message_text = elem.get("MsgContent", {}).get("Text")
                break
    
    # 准备回复的消息，回复的消息也是一个json格式的数据，比如下面的格式
    # {
    #     "ActionStatus": "OK",
    #     "ErrorInfo": "",
    #     "ErrorCode": 0, // 必须为0，只有这样，变动之后的消息才能正常下发
    #     "MsgBody": [ // App 变动之后的消息，如果没有，则默认使用用户发送的消息
    #         {
    #             "MsgType": "TIMTextElem", // 文本
    #             "MsgContent": {
    #                 "Text": "red packet"
    #             }
    #         },
    #         {
    #             "MsgType": "TIMCustomElem", // 自定义消息
    #             "MsgContent": {
    #                 "Desc": " CustomElement.MemberLevel ", // 描述
    #                 "Data": " LV1" // 数据
    #             }
    #         }
    #     ],
    #     "CloudCustomData": "your new cloud custom data" // 消息自定义数据
    # }
    response_json = {
        "ActionStatus": "OK",
        "ErrorInfo": "",
        "ErrorCode": 0,  
    }
    #即将根据from_account-to_account生成一个key，用来保存和维护不同人之间的聊天记录
    from_account = data.get("From_Account")
    to_account = data.get("To_Account")
    key = min(from_account, to_account) + '-' + max(from_account, to_account)

    # 根据不同的callback_command_param来处理不同的逻辑，共有两种情况：
    ###############################################################
    # 第一种：callback_command_param == "C2C.CallbackAfterSendMsg"，表示当一条消息已经被发送完成了
    # 此时我们要做的处理是调用大模型来判断相关的聊天记录是否有诈骗嫌疑，根据诈骗嫌疑的程度（1-5）进行不同的处理
    # 嫌疑程度1-2：不做处理
    # 嫌疑程度3：在后续对话中进行提醒，将AI返回的reasoning和advice加入notice变量，以便在后续对话中提醒用户
    # 嫌疑程度4-5：通过管理员账号直接发送消息给用户，提醒用户注意
    ###############################################################
    # 第二种：callback_command_param == "C2C.CallbackBeforeSendMsg"，表示当一条消息即将被发送
    # 此时我们要做的处理是根据前面的消息内容，来决定是否在前面的消息内容中加入notice的内容
    # 如果前一次消息的可疑程度大于等于3，表示用户回复了可疑消息，那就在这次的消息内容中加入notice的内容，以提醒用户谨防诈骗
    ###############################################################
    if callback_command_param == "C2C.CallbackAfterSendMsg":
        #下面是第一种情况的处理逻辑
        # 1. 按照from_account-to_account保存聊天记录
        add_message(from_account, to_account, message_text)
        
        # 2. 调用AI模型来判断聊天记录是否有诈骗嫌疑
        # 将最近的若干条消息传递给AI模型，AI模型会返回一个如下面格式的数据ai_response，包括可疑程度，原因和建议
        #```json 
        # { 
        #   "suspect_level": 5, 
        #   "reasoning": "对方自称银行职员要求提供身份证号和验证码，这是敏感信息，且未明确说明账户异常的具体情况。", 
        #   "advice": "立即挂断电话，不要提供任何信息，联系银行官方客服号码核实情况。" } 
        # ```        
        msg = conversations[key].get_lastmessages()
        print(f"[Request to AI]{msg}")
        ai_response = ai.check_suspect_level(msg)
        print(f"[AI Response] {ai_response}")

        # 3. 将AI返回的内容前后的```json去掉，然后转换成json格式，以便后续处理
        def remove_codeblock_markers(text):
            # Remove the starting marker "```json"
            if text.startswith("```json"):
                text = text[len("```json"):].lstrip()
            # Remove the ending marker "```"
            if text.endswith("```"):
                text = text[:-3].rstrip()
            return text        
        
        data = json.loads(remove_codeblock_markers(ai_response))
        # 4. 判断AI返回的内容，根据不同的可疑程度进行不同的处理
        if data:
            suspect_level = data.get("rating")
            # 如果可疑程度大于等于3（表示有诈骗可能但不确定，并需要在后续对话中进行）
            # 处理的方式是，如果用户回复了该消息，那就在后面的消息中自动加入notice的内容来进行提醒
            # notice的内容是AI返回的reasoning和advice
            if suspect_level and suspect_level >= 3:
                notice = f"---\n注意：{data.get("reasoning")}\n可疑程度：{suspect_level}\n建议：{data.get("advice")}\n---"

                # 5. 如果可疑程度大于等于4（表示有诈骗的可能性非常大，且已经开始获取私人信息，引诱点击可疑链接或要求转账等）
                # 处理的方式是，通过管理员账号发送消息给用户，提醒用户注意
                # 采用了腾讯云的IM服务，通过REST API来发送消息，相关文档在https://cloud.tencent.com/document/product/269/2282
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
        #下面是第二种情况的处理逻辑
        # 1. 如果前一次消息的可疑程度大于等于3，表示用户回复了可疑消息，那就在这次的消息内容中插入notice的内容，以提醒用户谨防诈骗
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

    #返回response_json给聊天服务器            
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