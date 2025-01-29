import time

def stream_output(output):

    for word in output:
        yield word
        time.sleep(0.005)