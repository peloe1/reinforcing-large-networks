import pandas as pd
import json
from typing import List, Dict, Any
import os

def load_trains_from_json(file_path: str) -> List[Dict]:
    """
    Load train data from JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        List of train dictionaries
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    # Handle different possible JSON structures
    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and 'trains' in data:
        return data['trains']
    else:
        raise ValueError("Unexpected JSON structure. Expected list of trains or dict with 'trains' key.")

def filter_trains_by_stations(trains_data: List[Dict], target_stations: List[str]) -> List[Dict]:
    """
    Filter trains that pass through, stop at, or start from the specified stations.
    
    Args:
        trains_data: List of train dictionaries
        target_stations: List of station short codes to filter by
        
    Returns:
        List of filtered train dictionaries
    """
    filtered_trains = []
    
    for train in trains_data:
        # Skip cancelled trains if desired
        if train.get('cancelled', False):
            continue
            
        # Check if this train has any activity at target stations
        for row in train.get('timeTableRows', []):
            if row.get('stationShortCode') in target_stations:
                filtered_trains.append(train)
                break  # No need to check other rows for this train
    
    return filtered_trains

def create_detailed_analysis(filtered_trains: List[Dict], target_stations: List[str]) -> pd.DataFrame:
    """
    Create a detailed DataFrame showing train activities at target stations.
    
    Args:
        filtered_trains: List of filtered train dictionaries
        target_stations: List of target station codes
        
    Returns:
        DataFrame with detailed analysis
    """
    analysis_data = []
    
    for train in filtered_trains:
        train_number = train['trainNumber']
        train_type = train['trainType']
        departure_date = train['departureDate']
        
        # Find all timetable rows for target stations
        for row in train.get('timeTableRows', []):
            station_code = row.get('stationShortCode')
            if station_code in target_stations:
                analysis_data.append({
                    'trainNumber': train_number,
                    'trainType': train_type,
                    'departureDate': departure_date,
                    'stationShortCode': station_code,
                    'eventType': row.get('type'),  # ARRIVAL or DEPARTURE
                    'scheduledTime': row.get('scheduledTime'),
                    'actualTime': row.get('actualTime'),
                    'liveEstimateTime': row.get('liveEstimateTime'),
                    'trainStopping': row.get('trainStopping', False),
                    'commercialStop': row.get('commercialStop', False),
                    'cancelled': row.get('cancelled', False),
                    'differenceInMinutes': row.get('differenceInMinutes', 0),
                    'commercialTrack': row.get('commercialTrack', '')
                })
    
    return pd.DataFrame(analysis_data)

def generate_summary_statistics(analysis_df: pd.DataFrame, filtered_trains: List[Dict]) -> Dict:
    """
    Generate summary statistics for the filtered data.
    
    Args:
        analysis_df: Detailed analysis DataFrame
        filtered_trains: List of filtered trains
        
    Returns:
        Dictionary with summary statistics
    """
    if analysis_df.empty:
        return {}
    
    summary = {
        'total_trains': len(filtered_trains),
        'total_events': len(analysis_df),
        'unique_stations_visited': analysis_df['stationShortCode'].nunique(),
        'trains_by_type': analysis_df.groupby('trainType')['trainNumber'].nunique().to_dict(),
        'events_by_station': analysis_df['stationShortCode'].value_counts().to_dict(),
        'events_by_type': analysis_df['eventType'].value_counts().to_dict(),
        'stopping_trains': analysis_df[analysis_df['trainStopping']]['trainNumber'].nunique(),
        'commercial_stops': analysis_df[analysis_df['commercialStop']]['trainNumber'].nunique()
    }
    
    return summary

def save_filtered_results(filtered_trains: List[Dict], output_file: str):
    """
    Save filtered results to a JSON file.
    
    Args:
        filtered_trains: List of filtered train dictionaries
        output_file: Path for output JSON file
    """

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(filtered_trains, file, indent=2, ensure_ascii=False)

# Main execution
def main():
    # Configuration
    # Taipale, Lapinlahti, Alapitkä, Siilinjärvi, Sänkimäki, Kinahmi, Juankoski, Toivala, Sorsasalo, Kuopio
    TARGET_STATIONS = ["TE", "LNA", "APT", "SIJ", "SKM", "KNH", "JKI", "TOI", "SOR", "KUO"]
    file_paths = []
    output_paths = []
    for i in range(1, 31+1):
        if i < 10:
            file_paths.append("data/raaka_data/2024-01/2024-01-0" + str(i) + "_trains.json")
            output_paths.append("data/juna_data/2024-01/2024-01-0" + str(i) + "_trains.json")
        else:
            file_paths.append("data/raaka_data/2024-01/2024-01-" + str(i) + "_trains.json")
            output_paths.append("data/juna_data/2024-01/2024-01-" + str(i) + "_trains.json")

    for JSON_FILE_PATH, OUTPUT_FILE in zip(file_paths, output_paths):
        try:
            # 1. Load data from JSON file
            print("Loading data from JSON file...")
            trains_data = load_trains_from_json(JSON_FILE_PATH)
            print(f"Loaded {len(trains_data)} trains from {JSON_FILE_PATH}")
            
            # 2. Filter trains for target stations
            print(f"Filtering trains for stations: {TARGET_STATIONS}")
            filtered_trains = filter_trains_by_stations(trains_data, TARGET_STATIONS)
            print(f"Found {len(filtered_trains)} trains that pass through target stations")
            
            if not filtered_trains:
                print("No trains found for the specified stations.")
                return
            
            # 3. Create detailed analysis
            print("Creating detailed analysis...")
            analysis_df = create_detailed_analysis(filtered_trains, TARGET_STATIONS)
            
            # 4. Generate summary statistics
            summary = generate_summary_statistics(analysis_df, filtered_trains)
            
            # 5. Display results
            print("\n" + "="*50)
            print("SUMMARY RESULTS")
            print("="*50)
            print(f"Total trains found: {summary['total_trains']}")
            print(f"Total station events: {summary['total_events']}")
            print(f"Unique stations visited: {summary['unique_stations_visited']}")
            print(f"Trains that actually stop: {summary['stopping_trains']}")
            print(f"Trains with commercial stops: {summary['commercial_stops']}")
            
            print("\nTrains by type:")
            for train_type, count in summary['trains_by_type'].items():
                print(f"  {train_type}: {count} trains")
            
            print("\nEvents by station:")
            for station, count in summary['events_by_station'].items():
                print(f"  {station}: {count} events")
            
            print("\nEvents by type:")
            for event_type, count in summary['events_by_type'].items():
                print(f"  {event_type}: {count} events")
            
            # 6. Show detailed data
            print("\n" + "="*50)
            print("DETAILED SCHEDULE AT TARGET STATIONS")
            print("="*50)
            
            # Display first 20 rows of detailed data
            display_columns = ['trainNumber', 'trainType', 'stationShortCode', 'eventType', 
                            'scheduledTime', 'trainStopping', 'commercialStop', 'differenceInMinutes']
            
            pd.set_option('display.width', None)
            pd.set_option('display.max_columns', None)
            print(analysis_df[display_columns].sort_values(['trainNumber', 'scheduledTime']).head(20).to_string(index=False))
            
            # 7. Save filtered results
            save_filtered_results(filtered_trains, OUTPUT_FILE)
            print(f"\nFiltered results saved to: {OUTPUT_FILE}")
            
            # 8. Additional analysis: Show trains with most station visits
            print("\n" + "="*50)
            print("TRAINS WITH MOST STATION VISITS")
            print("="*50)
            station_visits = analysis_df.groupby('trainNumber').agg({
                'stationShortCode': 'nunique',
                'trainType': 'first'
            }).sort_values('stationShortCode', ascending=False)
            
            print(station_visits.head(10).to_string())
            
        except FileNotFoundError:
            print(f"Error: File {JSON_FILE_PATH} not found.")
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON format in {JSON_FILE_PATH}.")
        except Exception as e:
            print(f"Error: {e}")

# Alternative: Simple one-liner approach for quick analysis
def quick_analysis(json_file_path: str, target_stations: List[str]):
    """
    Quick analysis function for immediate results.
    """
    # Load data
    with open(json_file_path, 'r') as file:
        trains_data = json.load(file)
    
    # Simple filtering
    filtered_trains = []
    for train in trains_data:
        if any(row.get('stationShortCode') in target_stations 
               for row in train.get('timeTableRows', [])):
            filtered_trains.append(train)
    
    print(f"Found {len(filtered_trains)} trains for stations {target_stations}")
    
    # Quick summary
    station_counts = {}
    for train in filtered_trains:
        for row in train['timeTableRows']:
            station = row.get('stationShortCode')
            if station in target_stations:
                station_counts[station] = station_counts.get(station, 0) + 1
    
    print("Events per station:")
    for station, count in sorted(station_counts.items()):
        print(f"  {station}: {count}")

# Run the main function
if __name__ == "__main__":
    main()
    
    # For quick analysis, uncomment below:
    # quick_analysis("trains_data.json", ["TE", "LNA", "APT", "SIJ", "SKM", "KNH", "JKI", "TOI", "SOR", "KUO"])