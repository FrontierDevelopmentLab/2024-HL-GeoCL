# Near Real-Time (NRT) Data Collection and Processing

## What is Near Real-Time (NRT) Data?

Near real-time (NRT) data refers to data that is collected, processed, and made available almost immediately after acquisition. While true real-time data implies zero latency, NRT data typically has a slight delay, often in the range of seconds to minutes, which is acceptable for most applications. NRT data is crucial for applications requiring timely updates, such as weather monitoring, financial trading, and environmental monitoring.

## How Google Services Aid in Scraping Data

Google Cloud Platform (GCP) offers several services that facilitate the collection, processing, and storage of NRT data. The primary services used in our setup include:

1. **Google Cloud Scheduler**: A fully managed cron job service that allows you to automate the execution of tasks at regular intervals. It can trigger HTTP endpoints or Google Cloud Functions to perform scheduled tasks, such as scraping data from external APIs.
   
2. **Google Cloud Functions**: A serverless execution environment for building and connecting cloud services. Cloud Functions can be triggered by Cloud Scheduler to execute specific tasks, like fetching NRT data from external sources and storing it in a database.

3. **InfluxDB**: A time-series database optimized for high-write throughput and query performance, making it ideal for storing and querying NRT data.

### Data Collection Frequency

Our system scrapes data from various sources at different intervals to ensure the availability of up-to-date information:

- **ACE and DSCOVR**: Every 5 minutes
- **Geomagnetic Indices**:
  - **Kp Index**: Every 3 hours
  - **Hp30 and ap30 Indices**: Every 30 minutes
  - **Fadj Index**: Every 24 hours

### Data Ingestion Pipeline

1. **Scheduling with Google Cloud Scheduler**:
   - Configure Cloud Scheduler to trigger scraping tasks at the specified intervals.
   - Cloud Scheduler can be set up to call specific Google Cloud Functions to perform the data scraping.

2. **Scraping with Google Cloud Functions**:
   - Each Cloud Function is designed to scrape data from specific sources based on the schedule.
   - For example, one function scrapes ACE and DSCOVR data every 5 minutes, while another scrapes Kp index data every 3 hours.
   - The functions parse the scraped data and format it appropriately.

3. **Storing Data in InfluxDB**:
   - The scraped and formatted data is then written to InfluxDB.
   - Each dataset is stored in its respective measurement within the InfluxDB database.
   - This allows for efficient querying and analysis of historical and NRT data.
