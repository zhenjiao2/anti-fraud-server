class Conversation:
    # 表示两个账户之间对话的类。
    # 属性
    # ----------
    # from_account : str
    #     发起对话的账户。
    # to_account : str
    #     接收对话的账户。
    # messages : list
    #     存储对话消息的列表。
    # 方法
    # -------
    # __init__(from_account, to_account, message=None):
    #     使用给定的账户和可选的初始消息初始化对话。
    # add_message(who, message):
    #     从指定账户添加消息到对话中。
    # get_lastmessages(count=3):
    #     返回对话中的最后 'count' 条消息。

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