from dotenv import load_dotenv
import os
# from tkinter import Tk, Button
from concurrent.futures import ThreadPoolExecutor

load_dotenv()

# print(os.getenv('GOVEE_API_KEY'))

import requests

url = 'https://developer-api.govee.com/v1/devices/control'
headers = {
    'Content-Type': 'application/json',
    'Govee-API-Key': os.getenv('GOVEE_API_KEY')  # Replace 'xxxx' with your actual API key
}

devices = [
    {'mac': os.getenv('FAN_LIGHT_1_MAC'), 'model': 'H6008'},
    {'mac': os.getenv('FAN_LIGHT_2_MAC'), 'model': 'H6008'}
]

# Track the bulb's current state
bulb_state = False  # False for off, True for on

def toggle_bulb(device, state):

    data = {
        'device': device['mac'],
        'model': device['model'],
        'cmd': {
            'name': 'turn',
            'value': state
        }
    }

    response = requests.put(url, headers=headers, json=data)
    # print(response.status_code, response.text)

def bulb_color(device, r, g, b):

    data = {
        'device': device['mac'],
        'model': device['model'],
        'cmd': {
            'name': 'color',
            'value': {
                'r': r,
                'g': g,
                'b': b
            }
        }
    }

    response = requests.put(url, headers=headers, json=data)
    # print(response.status_code, response.text)

def toggle_all_bulbs():
    global bulb_state
    bulb_state = not bulb_state
    state = 'on' if bulb_state else 'off'
    
    with ThreadPoolExecutor() as executor:
        executor.map(lambda device: toggle_bulb(device, state), devices)
    
    # button.config(text=f"Turn {'Off' if bulb_state else 'On'}")

# # Create the GUI
# root = Tk()
# root.title("Govee Bulb Controller")

# # Add a button to toggle the bulb
# button = Button(root, text="Turn On", command=toggle_all_bulbs, width=20, height=2)
# button.pack(pady=20)

# # Run the application
# root.mainloop()
