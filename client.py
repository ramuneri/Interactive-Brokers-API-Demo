from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from threading import Thread
import time

class Test(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)

    def error(self, reqId, errorCode, errorString):
        print("ERR", errorCode, errorString)

    def nextValidId(self, orderId):
        print("CONNECTED", orderId)

app = Test()
app.connect("127.0.0.1", 4001, clientId=10)
Thread(target=app.run, daemon=True).start()

time.sleep(3)
app.reqCurrentTime()  # <--- SIMPLE, SUPPORTED CALL NO MATTER WHAT
time.sleep(3)
