import requests
from bs4 import BeautifulSoup
import csv
from datetime import datetime, timedelta

# Function to scrape data for a specific date
def scrape_data(formatted_date):
    url = f"https://www.delhisldc.org/Loaddata.aspx?mode={formatted_date}"
    print(f"Scraping data for {formatted_date}...")
    
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for 4xx/5xx errors
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table')

        if table:
            rows = table.find_all('tr')
            row_data = []
            for row in rows:
                cols = row.find_all('td')
                cols = [col.text.strip() for col in cols]
                # Ensure the row has exactly 7 relevant columns to match the desired format
                if len(cols) >= 7 and "TIMESLOT" not in cols[0]:  # Skip header rows
                    # Append date and the required load columns in correct format
                    row_data.append([formatted_date] + cols[:7])
            return row_data  # Return the rows for this date
        else:
            print(f"No data found for {formatted_date}")
            return None

    except requests.exceptions.RequestException as e:
        print(f"Failed to fetch data for {formatted_date}: {e}")
        return None

# Main function to handle sequential scraping
def scrape_sequentially(start_date, end_date):
    all_data = []
    current_date = start_date

    # Loop through each day and scrape data sequentially
    while current_date <= end_date:
        formatted_date = current_date.strftime("%d/%m/%Y")
        result = scrape_data(formatted_date)
        if result:
            all_data.extend(result)  # Add the results of each date to the main list
        current_date += timedelta(days=1)

    return all_data

# Start and End Dates for a small range to test functionality
start_date = datetime(2022, 1, 1)
end_date = datetime(2022, 12, 31)

# Write the scraped data to a CSV file
output_file = 'delhi_load_data_sample_non_parallel.csv'
with open(output_file, 'w', newline='') as file:
    writer = csv.writer(file)
    # Write the header to match the required format
    writer.writerow(["Date", "TIMESLOT", "DELHI", "BRPL", "BYPL", "NDPL", "NDMC", "MES"])

    # Scrape data sequentially and collect it
    scraped_data = scrape_sequentially(start_date, end_date)

    # Write the collected data to the CSV file
    for row in scraped_data:
        writer.writerow(row)

print("Data scraping completed sequentially! Check the sample output file for verification.")
