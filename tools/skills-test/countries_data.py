import csv

# Set headers
headers = ["Country", "Population", "Area (sq km)", "GDP (trillion USD)"]

# Country data from search results
countries_data = [
    ["China", "1,425,722,992", "9,596,961", "17.89"],
    ["India", "1,426,711,933", "3,287,263", "3.75"],
    ["United States", "335,000,000", "9,833,517", "27.36"],
    ["Indonesia", "279,118,705", "1,904,569", "1.31"],
    ["Pakistan", "240,485,658", "881,912", "0.376"],
    ["Brazil", "216,422,446", "8,515,767", "2.08"],
    ["Nigeria", "223,804,632", "923,768", "0.477"],
    ["Bangladesh", "172,954,319", "147,570", "0.465"],
    ["Russia", "145,478,097", "17,098,242", "2.13"],
    ["Mexico", "140,760,000", "1,964,375", "1.74"],
    ["Japan", "125,000,000", "377,975", "4.24"],
    ["Germany", "83,200,000", "357,114", "4.5"],
    ["United Kingdom", "67,330,000", "243,610", "3.07"],
    ["France", "68,040,000", "551,695", "3.05"],
    ["Italy", "59,000,000", "301,338", "2.19"],
    ["Canada", "39,500,000", "9,984,670", "2.08"],
    ["Korea", "51,740,000", "100,210", "1.81"],
    ["Australia", "26,300,000", "7,692,024", "1.7"],
    ["Spain", "47,500,000", "505,990", "1.41"],
    ["Sweden", "10,170,000", "450,295", "0.669"]
]

# Create CSV file
with open('world_countries_data.csv', 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(headers)
    writer.writerows(countries_data)

print("CSV file 'world_countries_data.csv' has been created successfully!")
print("You can open this file in Excel or any spreadsheet application.")
