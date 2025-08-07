import datetime

# get current date and time 
def time_stamp_for_saving():
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")