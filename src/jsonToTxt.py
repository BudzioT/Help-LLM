import requests

# Get the needed info
url_name = input("Url Name: ")
path_name = input("File name: ")

# Get the URL file
response = requests.get(f"{url_name}")

# Convert the response to JSON
data = response.json()

# Open the text file
with open(f'../resources/data/{path_name}.txt', 'w') as f:
    # Iterate over each item in the data
    for item in data:
        # Write the input and output to the text file
        f.write(item['input'] + '\n')
        f.write(item['output'] + '\n')