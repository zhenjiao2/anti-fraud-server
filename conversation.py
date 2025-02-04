class Conversation:

    def __init__(self, from_account, to_account, message=None):
        self.from_account = from_account
        self.to_account = to_account
        self.messages = []
        if message:
            self.messages.append(self.from_account + ": " + message)

    def add_message(self, who, message):
        self.messages.append(who + ": " + message)

    def get_lastmessages(self, count=3):
        return self.messages[-count:]