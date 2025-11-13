import pandas as pd
import json
from typing import List, Dict, Any
import os

TOTAL_TRAIN_COUNTER = 0

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


def find_path_between_stations(start: str, end: str, topology: Dict) -> List[str]:
    """
    Find the shortest path between two stations using BFS.
    """
    if start == end:
        return [start]
    
    queue = [(start, [start])]
    visited = set([start])
    
    while queue:
        current, path = queue.pop(0)
        
        for neighbor in topology.get(current, []):
            if neighbor == end:
                return path + [neighbor]
            
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    
    return []  # No path found (shouldn't happen in our connected network)

def filter_trains_by_stations(trains_data: List[Dict], target_stations: List[str]) -> List[Dict]:
    """
    Filter trains that pass through, stop at, or start from the specified stations.
    Also reconstructs complete paths for each train.
    """
    filtered_trains = []
    
    for train in trains_data:
        # Skip cancelled trains if desired
        if train.get('cancelled', False):
            continue
            
        # Check if this train has any activity at target stations
        has_target_station = False
        for row in train.get('timeTableRows', []):
            station = row.get('stationShortCode')
            if station == "KUOT":
                station = "KUO"
            if station in target_stations:
                has_target_station = True
                break
        
        if has_target_station:
            # Reconstruct the complete path for this train
            reconstructed_path = reconstruct_complete_path(train.get('timeTableRows', []))
            train['reconstructedPath'] = reconstructed_path
            filtered_trains.append(train)
    
    return filtered_trains

def create_detailed_analysis(filtered_trains: List[Dict], target_stations: List[str]) -> pd.DataFrame:
    """
    Create a detailed DataFrame showing train activities at target stations.
    """
    analysis_data = []
    
    for train in filtered_trains:
        train_number = train['trainNumber']
        train_type = train['trainType']
        departure_date = train['departureDate']
        
        # Use reconstructed path for analysis
        reconstructed_path = train.get('reconstructedPath', [])
        
        for station_code in reconstructed_path:
            if station_code in target_stations:
                # Find the corresponding timetable row for timing information
                timetable_row = None
                for row in train.get('timeTableRows', []):
                    if row.get('stationShortCode') == station_code or (row.get('stationShortCode') == "KUOT" and station_code == "KUO"):
                        timetable_row = row
                        break
                
                analysis_data.append({
                    'trainNumber': train_number,
                    'trainType': train_type,
                    'departureDate': departure_date,
                    'stationShortCode': station_code,
                    'eventType': timetable_row.get('type') if timetable_row else 'INFERRED',
                    'scheduledTime': timetable_row.get('scheduledTime') if timetable_row else None,
                    'actualTime': timetable_row.get('actualTime') if timetable_row else None,
                    'liveEstimateTime': timetable_row.get('liveEstimateTime') if timetable_row else None,
                    'trainStopping': timetable_row.get('trainStopping', False) if timetable_row else False,
                    'commercialStop': timetable_row.get('commercialStop', False) if timetable_row else False,
                    'cancelled': timetable_row.get('cancelled', False) if timetable_row else False,
                    'differenceInMinutes': timetable_row.get('differenceInMinutes', 0) if timetable_row else 0,
                    'commercialTrack': timetable_row.get('commercialTrack', '') if timetable_row else '',
                    'pathPosition': reconstructed_path.index(station_code)
                })
    
    return pd.DataFrame(analysis_data)

def save_filtered_results(filtered_trains: List[Dict], output_file: str):
    """
    Save filtered results to a JSON file, including reconstructed paths.
    """
    global TOTAL_TRAIN_COUNTER
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Create a clean version for output (remove large binary data if any)
    output_data = []
    for train in filtered_trains:
        clean_train = {k: v for k, v in train.items() if k != 'reconstructedPath'}
        clean_train['reconstructedPath'] = train.get('reconstructedPath', [])
        output_data.append(clean_train)
    
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(output_data, file, indent=2, ensure_ascii=False)
    
    
    TOTAL_TRAIN_COUNTER += len(filtered_trains)


def reconstruct_complete_path(timetable_rows: List[Dict]) -> List[str]:
    """
    Reconstruct the complete station path from timetable rows,
    but only validate and fix the relevant subnetwork part.
    """
    # Only define topology for our relevant subnetwork
    RELEVANT_TOPOLOGY = {
        "KRM": ["KUO"],
        "KUO": ["KRM", "SOR"],
        "SOR": ["KUO", "TOI"],
        "TOI": ["SOR", "SIJ"],
        "SIJ": ["TOI", "APT", "SKM"],
        "APT": ["SIJ", "LNA"],
        "LNA": ["APT", "TE"],
        "TE": ["LNA", "OHM"],
        "OHM": ["TE"],
        "SKM": ["SIJ", "KNH"],
        "KNH": ["SKM", "JKI"],
        "JKI": ["KNH", "LUI"],
        "LUI": ["JKI"]
    }
    
    RELEVANT_STATIONS = set(RELEVANT_TOPOLOGY.keys())
    
    # Extract unique stations in order from timetable
    raw_stations = []
    last_station = None
    
    for row in timetable_rows:
        station = row.get('stationShortCode')
        if station and station != last_station:
            if station == "KUOT":
                station = "KUO"
            raw_stations.append(station)
            last_station = station
    
    if len(raw_stations) < 2:
        return raw_stations
    
    # Only reconstruct paths within our relevant subnetwork
    reconstructed_path = [raw_stations[0]]
    
    for i in range(1, len(raw_stations)):
        current_station = raw_stations[i]
        previous_station = reconstructed_path[-1]
        
        # Only validate if both stations are in our relevant subnetwork
        if previous_station in RELEVANT_STATIONS and current_station in RELEVANT_STATIONS:
            # Check if direct connection exists in our subnetwork
            if current_station in RELEVANT_TOPOLOGY.get(previous_station, []):
                reconstructed_path.append(current_station)
            else:
                # Find path between relevant stations
                path_segment = find_path_between_stations(previous_station, current_station, RELEVANT_TOPOLOGY)
                if path_segment:
                    reconstructed_path.extend(path_segment[1:])
                else:
                    # If no path found in our subnetwork, just add it (external connection)
                    reconstructed_path.append(current_station)
        else:
            # For stations outside our subnetwork, just trust the raw data
            reconstructed_path.append(current_station)
    
    return reconstructed_path

def validate_reconstructed_paths(filtered_trains: List[Dict]):
    """
    Validate that only the relevant subnetwork paths are physically possible.
    """
    RELEVANT_TOPOLOGY = {
        "KRM": ["KUO"], "KUO": ["KRM", "SOR"], "SOR": ["KUO", "TOI"],
        "TOI": ["SOR", "SIJ"], "SIJ": ["TOI", "APT", "SKM"], "APT": ["SIJ", "LNA"],
        "LNA": ["APT", "TE"], "TE": ["LNA", "OHM"], "OHM": ["TE"],
        "SKM": ["SIJ", "KNH"], "KNH": ["SKM", "JKI"], "JKI": ["KNH", "LUI"], "LUI": ["JKI"]
    }
    
    RELEVANT_STATIONS = set(RELEVANT_TOPOLOGY.keys())
    
    invalid_segments = []
    
    for train in filtered_trains:
        path = train.get('reconstructedPath', [])
        for i in range(len(path) - 1):
            current = path[i]
            next_station = path[i + 1]
            
            # Only validate if both stations are in our relevant subnetwork
            if current in RELEVANT_STATIONS and next_station in RELEVANT_STATIONS:
                if next_station not in RELEVANT_TOPOLOGY.get(current, []):
                    invalid_segments.append({
                        'trainNumber': train.get('trainNumber'),
                        'invalidSegment': f"{current} → {next_station}",
                        'fullPath': path
                    })
    
    if invalid_segments:
        error_message = f"Found {len(invalid_segments)} invalid path segments in relevant subnetwork:\n"
        for invalid in invalid_segments[:10]:
            error_message += f"  Train {invalid['trainNumber']}: {invalid['invalidSegment']} in path {invalid['fullPath']}\n"
        raise ValueError(error_message)
    
    return True

def get_total_trains_counter():
    global TOTAL_TRAIN_COUNTER
    return TOTAL_TRAIN_COUNTER

def reset_total_trains_counter():
    global TOTAL_TRAIN_COUNTER
    TOTAL_TRAIN_COUNTER = 0
    return

# Update the main function to handle exceptions properly
def main():
    reset_total_trains_counter()
    # Include ALL stations including terminals
    SUBNETWORK_STATIONS = {
        "KRM": "Kurkimäki", 
        "KUO": "Kuopio", 
        "SOR": "Sorsasalo", 
        "TOI": "Toivala", 
        "SIJ": "Siilinjärvi", 
        "APT": "Alapitkä",
        "LNA": "Lapinlahti", 
        "TE": "Taipale", 
        "OHM": "Ohenmäki", 
        "SKM": "Sänkimäki", 
        "KNH": "Kinahmi", 
        "JKI": "Juankoski", 
        "LUI": "Luikonlahti"
    }
    
    TARGET_STATIONS = list(SUBNETWORK_STATIONS.keys())
    
    monthIndexes = ["0" + str(i) for i in range(1, 10)] + ["10", "11", "12"]
    for month in monthIndexes:
        file_paths = []
        output_paths = []
        for i in range(1, 31+1):
            if i < 10:
                file_paths.append("data/raaka_data/2024-" + month + "/2024-" + month + "-0" + str(i) + "_trains.json")
                output_paths.append("data/juna_data/2024-" + month + "/2024-" + month + "-0" + str(i) + "_trains.json")
            else:
                file_paths.append("data/raaka_data/2024-" + month + "/2024-" + month + "-" + str(i) + "_trains.json")
                output_paths.append("data/juna_data/2024-" + month + "/2024-" + month + "-" + str(i) + "_trains.json")

        for JSON_FILE_PATH, OUTPUT_FILE in zip(file_paths, output_paths):
            try:
                # 1. Load data from JSON file
                print(f"Loading data from {JSON_FILE_PATH}...")
                trains_data = load_trains_from_json(JSON_FILE_PATH)
                print(f"Loaded {len(trains_data)} trains")
                
                # 2. Filter trains for target stations and reconstruct paths
                print("Filtering trains and reconstructing paths...")
                filtered_trains = filter_trains_by_stations(trains_data, TARGET_STATIONS)
                print(f"Found {len(filtered_trains)} trains that pass through target stations")
                
                if not filtered_trains:
                    print("No trains found for the specified stations.")
                    continue
                
                # 3. Validate reconstructed paths - this will throw exception if invalid
                print("Validating reconstructed paths...")
                validate_reconstructed_paths(filtered_trains)
                print("✅ All reconstructed paths are valid")
                
                # 4. Save filtered results with reconstructed paths
                save_filtered_results(filtered_trains, OUTPUT_FILE)
                print(f"✅ Saved {len(filtered_trains)} trains with reconstructed paths to: {OUTPUT_FILE}")
                
            except ValueError as e:
                print(f"❌ CRITICAL ERROR in {JSON_FILE_PATH}: {e}")
                # You can choose to stop execution or continue with next file
                # raise e  # Uncomment to stop execution on first error
            except FileNotFoundError:
                print(f"❌ File not found: {JSON_FILE_PATH}")
            except Exception as e:
                print(f"❌ Unexpected error processing {JSON_FILE_PATH}: {e}")

    print("Total trains written: ", get_total_trains_counter())

# Run the main function
if __name__ == "__main__":
    main()
    
    # For quick analysis, uncomment below:
    # quick_analysis("trains_data.json", ["TE", "LNA", "APT", "SIJ", "SKM", "KNH", "JKI", "TOI", "SOR", "KUO"])