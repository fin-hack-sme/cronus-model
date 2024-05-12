from datetime import datetime
from pprint import pprint
import random


def info(message):
    print(f"{datetime.now()}: {message}")


def pinfo(message):
    print(f"{datetime.now()}: data as below")
    pprint(message)


def generate_hex_number_with_date_prefix():
    # Get the current date as a prefix
    current_date = datetime.now().strftime("%Y%m%d%H%M%S")

    # Generate a random 6-digit hexadecimal number
    hex_number = ''.join(random.choice('0123456789ABCDEF') for _ in range(6))

    # Combine the date prefix and the hex number
    result = f"{current_date}-{hex_number}"

    return result
