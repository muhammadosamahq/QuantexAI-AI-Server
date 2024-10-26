import requests

def test_start_detection():
    url = "http://127.0.0.1:8000/start-detection"
    try:
        response = requests.get(url)

        if response.status_code == 200:
            print("Test Passed!")
            print("Response:", response.json())
        else:
            print("Test Failed!")
            print(f"Status Code: {response.status_code}")
            print("Response:", response.text)
    
    except requests.exceptions.RequestException as e:
        print("Test Failed!")
        print("Error:", e)

if __name__ == "__main__":
    test_start_detection()
