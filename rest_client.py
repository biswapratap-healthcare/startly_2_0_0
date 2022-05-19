import time
import requests


def test_search_speed():
    start_time = time.time()
    r = requests.post("http://a07588430af2e480f8b03d43a5e9c4e5-1713502582.ap-south-1.elb.amazonaws.com:8000/test_search_speed?style=Beige_Soft_Lifestyle&base_threshold=100&filter_threshold=5")
    # r = requests.post("http://65.0.127.236:8000/test_search_speed?style=Beige_Soft_Lifestyle&base_threshold=100&filter_threshold=5")
    search_time = time.time() - start_time
    print('Search Time = ', search_time)


if __name__ == "__main__":
    test_search_speed()
