## ACE, DSCOVR, and Geomagnetic Indices Data

### ACE Data
The Advanced Composition Explorer (ACE) data starts from the year 2001. ACE monitors the solar wind and interstellar particles to provide continuous real-time data, which is crucial for space weather forecasting.

### DSCOVR Data
The Deep Space Climate Observatory (DSCOVR) provides critical measurements of solar wind and magnetic field. The data collection starts from 2016. DSCOVR plays a key role in providing early warnings of potentially harmful space weather events.

### Geomagnetic Indices Data
Geomagnetic indices such as Hp30, ap30, Kp, and Fadj also have data starting from 2016. These indices are essential for understanding the geomagnetic conditions and their impact on Earth's magnetosphere.

### Historical Data Migration
All historical data from ACE, DSCOVR, and geomagnetic indices have been migrated to InfluxDB, ensuring a comprehensive archive of space weather data until the end of the Summer 2024 program. The data migration ensures that researchers and analysts can access a continuous timeline of space weather events and their corresponding indices.

### InfluxDB Integration
Our InfluxDB instance contains all the historical data from these sources up until the Summer 2024 program's conclusion. The InfluxDB setup allows for efficient querying and analysis of time-series data, enabling detailed studies and real-time monitoring of space weather phenomena.

### Data Scraping Scripts
The scripts in this directory are designed to scrape data from specific dates, ensuring that the database remains up-to-date with the latest observations. These scripts automate the data collection process, pulling the required information from various sources and inserting it into the InfluxDB database.
