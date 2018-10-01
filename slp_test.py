#  /usr/bin/python
# -*- coding: utf-8 -*-
import time


def keyinterrupt():
    try:
        time.sleep(20)
    except KeyboardInterrupt as ke:
        print(ke)
        pass

    print("all done")

