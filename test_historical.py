from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from threading import Thread
import time


class TestAPI(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)

    def error(self, reqId, errorCode, errorString):
        print("ERR", errorCode, errorString)

    def nextValidId(self, orderId):
        print("CONNECTED", orderId)

    def historicalData(self, reqId, bar):
        print("BAR:", bar.date, bar.close)

    def historicalDataEnd(self, reqId, start, end):
        print("HIST END")


def main():
    app = TestAPI()
    app.connect("127.0.0.1", 4001, clientId=2)

    Thread(target=app.run, daemon=True).start()
    time.sleep(2)

    # Create EUR/USD contract
    contract = Contract()
    contract.symbol = "EUR"
    contract.secType = "CASH"
    contract.exchange = "IDEALPRO"
    contract.currency = "USD"

    # Request last 1 minute of midpoint data
    app.reqHistoricalData(
        reqId=1,
        contract=contract,
        endDateTime="",
        durationStr="60 S",
        barSizeSetting="30 secs",
        whatToShow="MIDPOINT",
        useRTH=0,
        formatDate=1,
        keepUpToDate=False,
        chartOptions=[]
    )

    time.sleep(5)
    app.disconnect()


main()
