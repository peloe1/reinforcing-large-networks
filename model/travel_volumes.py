import json
from collections import defaultdict
import os

# Updated parameters with your real stations
SUBNETWORK_STATIONS = {
    "KRM": "Kurkimäki",        # Southern terminal area
    #"KUOT": "Kuopio varikkotms",
    "KUO": "Kuopio",           # Internal station
    "SOR": "Sorsasalo", 
    "TOI": "Toivala",
    "SIJ": "Siilinjärvi",
    "APT": "Alapitkä",
    "LNA": "Lapinlahti",       # Internal station
    "TE": "Taipale",           # Northern area
    "OHM": "Ohenmäki",         # Northern terminal  
    "SKM": "Sänkimäki",        # Eastern branch
    "KNH": "Kinahmi",          # Eastern branch  
    "JKI": "Juankoski",        # Internal station
    "LUI": "Luikonlahti"       # Eastern terminal
}

SUBNETWORK_TERMINAL_PAIRS = {
    "KUO": [("KRM", "KUO"), ("KRM", "SOR"), ("KUO", "SOR")],
    "SOR": [("KUO", "SOR"), ("KUO", "TOI"), ("SOR", "TOI")],
    "TOI": [("TOI", "SOR"), ("TOI", "SIJ"), ("SOR", "SIJ")],
    "SIJ": [("TOI", "SIJ"), ("TOI", "SKM"), ("TOI", "APT"), ("SIJ", "SKM"), ("SIJ", "APT")],
    "APT": [("SIJ", "APT"), ("SIJ", "LNA"), ("APT", "LNA")],
    "LNA": [("APT", "LNA"), ("APT", "TE"), ("LNA", "TE")],
    "TE": [("LNA", "TE"), ("LNA", "OHM"), ("TE", "OHM")],
    "SKM": [("SIJ", "SKM"), ("SIJ", "KNH"), ("SKM", "KNH")],
    "KNH": [("SKM", "KNH"), ("SKM", "JKI"), ("KNH", "JKI")],
    "JKI": [("KNH", "JKI"), ("KNH", "LUI"), ("JKI", "LUI")]
}

# Terminal nodes
TERMINALS = {
    "south": "KRM",
    "north": "OHM", 
    "east": "LUI"
}

TERMINALS = list(SUBNETWORK_STATIONS.keys())

# All stations that should be tracked (subnetwork + terminals)
ALL_TRACKED_STATIONS = set(SUBNETWORK_STATIONS.keys())

def derive_subnetwork_volumes(path_frequencies):
    """
    Derive travel volumes for each subnetwork's terminal pairs based on which trains
    actually pass through that subnetwork via specific entry/exit points.
    Treats terminal pairs as unordered (A,B) == (B,A)
    """
    SUBNETWORK_TERMINAL_PAIRS = {
        "KUO": [("KRM", "KUO"), ("KRM", "SOR"), ("KUO", "SOR")],
        "SOR": [("KUO", "SOR"), ("KUO", "TOI"), ("SOR", "TOI")],
        "TOI": [("TOI", "SOR"), ("TOI", "SIJ"), ("SOR", "SIJ")],
        "SIJ": [("TOI", "SIJ"), ("TOI", "SKM"), ("TOI", "APT"), ("SIJ", "SKM"), ("SIJ", "APT")],
        "APT": [("SIJ", "APT"), ("SIJ", "LNA"), ("APT", "LNA")],
        "LNA": [("APT", "LNA"), ("APT", "TE"), ("LNA", "TE")],
        "TE": [("LNA", "TE"), ("LNA", "OHM"), ("TE", "OHM")],
        "SKM": [("SIJ", "SKM"), ("SIJ", "KNH"), ("SKM", "KNH")],
        "KNH": [("SKM", "KNH"), ("SKM", "JKI"), ("KNH", "JKI")],
        "JKI": [("KNH", "JKI"), ("KNH", "LUI"), ("JKI", "LUI")]
    }
    
    SUBNETWORK_STATIONS = {
        "KUO": ["KRM", "KUO", "SOR"],
        "SOR": ["KUO", "SOR", "TOI"], 
        "TOI": ["SOR", "TOI", "SIJ"],
        "SIJ": ["TOI", "SIJ", "SKM", "APT"],
        "APT": ["SIJ", "APT", "LNA"],
        "LNA": ["APT", "LNA", "TE"],
        "TE": ["LNA", "TE", "OHM"],
        "SKM": ["SIJ", "SKM", "KNH"],
        "KNH": ["SKM", "KNH", "JKI"],
        "JKI": ["KNH", "JKI", "LUI"]
    }
    
    subnetwork_volumes = {}
    
    for subnetwork, terminal_pairs in SUBNETWORK_TERMINAL_PAIRS.items():
        terminal_pair_volumes = defaultdict(int)
        subnetwork_stations = set(SUBNETWORK_STATIONS[subnetwork])
        
        for path, frequency in path_frequencies.items():
            path_list = list(path)
            
            # Find which stations in this path belong to our subnetwork
            subnetwork_indices = []
            for i, station in enumerate(path_list):
                if station in subnetwork_stations:
                    subnetwork_indices.append((i, station))
            
            if len(subnetwork_indices) >= 2:
                # The entry point is the first subnetwork station
                entry_index, entry_station = subnetwork_indices[0]
                # The exit point is the last subnetwork station  
                exit_index, exit_station = subnetwork_indices[-1]
                
                # Create sorted tuple to make direction irrelevant
                terminal_pair = tuple(sorted([entry_station, exit_station]))
                
                # Check if this sorted terminal pair is valid for this subnetwork
                # We need to check if the sorted version exists in terminal_pairs
                sorted_terminal_pairs = [tuple(sorted(pair)) for pair in terminal_pairs]
                if terminal_pair in sorted_terminal_pairs:
                    terminal_pair_volumes[terminal_pair] += frequency
        
        subnetwork_volumes[subnetwork] = dict(terminal_pair_volumes)
    
    return subnetwork_volumes

def print_subnetwork_summary(subnetwork_volumes):
    """
    Print a summary for each subnetwork's terminal pair volumes.
    """
    if not subnetwork_volumes:
        print("No subnetwork volume data to display.")
        return
        
    for subnetwork, volumes in subnetwork_volumes.items():
        if volumes:
            print(f"\n=== {subnetwork} SUBNETWORK VOLUME SUMMARY ===")
            print("-" * 50)
            
            total_volume = 0
            for (from_station, to_station), volume in sorted(volumes.items(), key=lambda x: x[1], reverse=True):
                from_name = SUBNETWORK_STATIONS.get(from_station, from_station)
                to_name = SUBNETWORK_STATIONS.get(to_station, to_station)
                print(f"  {from_station} → {to_station}: {volume} trains  ({from_name} → {to_name})")
                total_volume += volume
            
            print(f"Total {subnetwork} subnetwork journeys: {total_volume}")

def save_subnetwork_volumes(subnetwork_volumes, output_dir):
    """
    Save separate volume files for each subnetwork.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for subnetwork, volumes in subnetwork_volumes.items():
        # Create filename for this subnetwork
        output_file = os.path.join(output_dir, f"{subnetwork}_volumes.json")
        
        # Convert tuple keys to strings for JSON
        output_dict = {}
        for key, value in volumes.items():
            key_str = str(key)
            output_dict[key_str] = value
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_dict, f, indent=2, ensure_ascii=False)
        
        print(f"Subnetwork {subnetwork} volumes saved to {output_file}")

def analyze_path_frequencies(train_data):
    """
    Use the reconstructed paths from data.py instead of analyzing paths here.
    """
    path_frequencies = defaultdict(int)
    
    # Get the reconstructed path that data.py already created
    reconstructed_path = train_data.get('reconstructedPath', [])
    
    # Filter to only include stations in our subnetwork
    subnetwork_path = [station for station in reconstructed_path if station in ALL_TRACKED_STATIONS]
    
    # If we have at least 2 subnetwork stations, track the path
    if len(subnetwork_path) >= 2:
        path_tuple = tuple(subnetwork_path)
        path_frequencies[path_tuple] += 1
        print(f"Train {train_data.get('trainNumber')} path: {' → '.join(subnetwork_path)}")
    
    return path_frequencies

def load_existing_path_frequencies(output_path_file):
    """
    Load existing path frequencies from JSON file if it exists.
    """
    if os.path.exists(output_path_file):
        try:
            with open(output_path_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # Convert string keys back to tuples
            path_frequencies = defaultdict(int)
            for key_str, value in data.items():
                # Convert string like "('KUO','SOR','TOI')" back to tuple
                key_str_clean = key_str.strip("()")
                parts = [part.strip().strip("'\"") for part in key_str_clean.split(",")]
                key_tuple = tuple(parts)
                path_frequencies[key_tuple] = value
            print(f"Loaded existing path frequencies from {output_path_file}")
            return path_frequencies
        except Exception as e:
            print(f"Error loading existing path frequencies: {e}. Starting fresh.")
            return defaultdict(int)
    else:
        print("No existing path frequency file found. Starting fresh.")
        return defaultdict(int)

def save_path_frequencies(path_frequencies, output_path_file):
    """
    Save path frequencies to JSON file.
    """
    os.makedirs(os.path.dirname(output_path_file), exist_ok=True)
    
    output_dict = {}
    for key, value in path_frequencies.items():
        key_str = str(key)
        output_dict[key_str] = value
    
    with open(output_path_file, 'w', encoding='utf-8') as f:
        json.dump(output_dict, f, indent=2, ensure_ascii=False)
    
    print(f"Path frequencies saved to {output_path_file}")

def print_path_summary(path_frequencies):
    """
    Print a summary of path frequencies.
    """
    if not path_frequencies:
        print("No path frequency data to display.")
        return
        
    print("\n=== PATH FREQUENCY SUMMARY ===")
    print("-" * 60)
    
    total_paths = 0
    for path, freq in sorted(path_frequencies.items(), key=lambda x: x[1], reverse=True):
        path_str = ' → '.join(path)
        print(f"  {path_str}: {freq} train(s)")
        total_paths += freq
    
    print(f"\nTotal paths tracked: {total_paths}")

def analyze_traffic_frequencies(train_data):
    """
    Analyze traffic frequencies - track ALL station pairs as terminals.
    """
    
    frequencies = defaultdict(int)
    
    # Get all timetable rows for this train
    rows = train_data.get('timeTableRows', [])
    
    # Find the actual first and last subnetwork stations in the entire journey
    first_subnetwork_station = None
    last_subnetwork_station = None
    
    for row in rows:
        station = row.get('stationShortCode', '')
        
        # Map KUOT to KUO
        if station == "KUOT":
            station = "KUO"
        
        if station in ALL_TRACKED_STATIONS:
            if first_subnetwork_station is None:
                first_subnetwork_station = station
            last_subnetwork_station = station
    
    # If no tracked stations, this train doesn't go through our subnetwork
    if first_subnetwork_station is None:
        return frequencies, False, None, None
    
    # Both entry and exit must be in our terminal list (all subnetwork stations)
    if first_subnetwork_station in TERMINALS and last_subnetwork_station in TERMINALS:
        frequencies[(first_subnetwork_station, last_subnetwork_station)] += 1
        print(f"Train {train_data.get('trainNumber')}: {first_subnetwork_station} → {last_subnetwork_station}")
        return frequencies, True, first_subnetwork_station, last_subnetwork_station
    
    return frequencies, True, first_subnetwork_station, last_subnetwork_station

def load_existing_frequencies(output_file):
    """
    Load existing frequencies from JSON file if it exists.
    Returns a defaultdict with the existing frequencies.
    """
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # Convert string keys back to tuples and create defaultdict
            frequencies = defaultdict(int)
            for key_str, value in data.items():
                # Convert string like "('KUO', 'SOR')" back to tuple
                key_str_clean = key_str.strip("()")
                parts = [part.strip().strip("'\"") for part in key_str_clean.split(",")]
                key_tuple = tuple(parts)
                frequencies[key_tuple] = value
            print(f"Loaded existing frequencies from {output_file}")
            return frequencies
        except Exception as e:
            print(f"Error loading existing frequencies: {e}. Starting fresh.")
            return defaultdict(int)
    else:
        print("No existing frequency file found. Starting fresh.")
        return defaultdict(int)

def save_frequencies(frequencies, output_file):
    """
    Save frequencies to JSON file.
    Convert tuple keys to strings for JSON compatibility.
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Convert defaultdict to regular dict with string keys
    output_dict = {}
    for key, value in frequencies.items():
        key_str = str(key)  # Convert tuple to string like "('KUO', 'SOR')"
        output_dict[key_str] = value
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_dict, f, indent=2, ensure_ascii=False)
    
    print(f"Frequencies saved to {output_file}")

def print_terminal_summary(frequencies):
    """
    Print a summary of station-to-station frequencies.
    """
    if not frequencies:
        print("No station-to-station frequency data to display.")
        return
        
    print("\n=== STATION-TO-STATION FREQUENCY SUMMARY ===")
    print("-" * 50)
    
    total_trains = 0
    for (from_station, to_station), freq in sorted(frequencies.items()):
        from_name = SUBNETWORK_STATIONS.get(from_station, from_station)
        to_name = SUBNETWORK_STATIONS.get(to_station, to_station)
        print(f"  {from_station} → {to_station}: {freq} train(s)  ({from_name} → {to_name})")
        total_trains += freq
    
    print(f"\nTotal station journeys: {total_trains}")

def process_trains_from_file(input_file, output_file):
    """
    Main function to process trains from JSON file and update frequencies.
    """
    # Load existing frequencies
    all_frequencies = load_existing_frequencies(output_file)
    
    # Load existing path frequencies
    output_path_file = output_file.replace('_volumes.json', '_path_volumes.json')
    all_path_frequencies = load_existing_path_frequencies(output_path_file)
    
    # Load train data
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            trains_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        return all_frequencies
    except Exception as e:
        print(f"Error loading train data: {e}")
        return all_frequencies
    
    # Handle both single train and list of trains
    if isinstance(trains_data, dict):
        trains_data = [trains_data]
    elif not isinstance(trains_data, list):
        print("Invalid data format: expected dict or list of dicts")
        return all_frequencies
    
    print(f"Processing {len(trains_data)} trains...")
    
    # Process each train
    processed_count = 0
    station_journeys_count = 0
    subnetwork_trains_count = 0
    missing_trains = []
    
    for train in trains_data:
        if isinstance(train, dict):
            frequencies, in_subnetwork, entry_station, exit_station = analyze_traffic_frequencies(train)
            path_frequencies = analyze_path_frequencies(train)
            
            if in_subnetwork:
                subnetwork_trains_count += 1
                if not frequencies:
                    missing_trains.append({
                        'train_number': train.get('trainNumber'),
                        'entry': entry_station,
                        'exit': exit_station
                    })
            
            # Increment the frequencies
            for station_pair, count in frequencies.items():
                all_frequencies[station_pair] += count
                station_journeys_count += count
            
            # Increment path frequencies
            for path, count in path_frequencies.items():
                all_path_frequencies[path] += count
            
            processed_count += 1
    
    print(f"\n=== DEBUG INFO ===")
    print(f"Total trains in dataset: {len(trains_data)}")
    print(f"Trains passing through subnetwork: {subnetwork_trains_count}")
    print(f"Station-to-station journeys: {station_journeys_count}")
    print(f"Successfully processed {processed_count} trains")
    
    # Print missing trains info
    if missing_trains:
        print(f"\n=== MISSING TRAINS (no station pair) ===")
        for train_info in missing_trains:
            print(f"Train {train_info['train_number']}: {train_info['entry']} → {train_info['exit']}")
    
    # Print station summary
    print_terminal_summary(all_frequencies)
    
    # Print path summary
    print_path_summary(all_path_frequencies)
    
    # Calculate and save subnetwork volumes - UPDATED THIS BLOCK
    subnetwork_volumes = derive_subnetwork_volumes(all_path_frequencies)
    print_subnetwork_summary(subnetwork_volumes)
    
    # Save subnetwork volumes to separate files - UPDATED THIS LINE
    output_subnetworks_dir = "data/travel_volumes/subnetwork_yearly"
    save_subnetwork_volumes(subnetwork_volumes, output_subnetworks_dir)
    
    # Save updated frequencies
    save_frequencies(all_frequencies, output_file)
    save_path_frequencies(all_path_frequencies, output_path_file)
    
    return all_frequencies
# Example usage
if __name__ == "__main__":
    monthIndexes = ["0" + str(i) for i in range(1, 10)] + ["10", "11", "12"]
    output_file = "data/travel_volumes/2024_volumes.json"
    for month in monthIndexes:
        inputs = []
        outputs = []

        for i in range(1, 31+1):
            #outputs.append("data/travel_volumes/2024-" + month + "_volumes.json")
            if i < 10:
                inputs.append("data/juna_data/2024-" + month + "/2024-" + month + "-0" + str(i) + "_trains.json")
                #outputs.append("data/travel_volumes/2024-" + month + "/2024-" + month + "-0"+ str(i) + "_trains.json")
            else:
                inputs.append("data/juna_data/2024-" + month + "/2024-" + month + "-" + str(i) + "_trains.json")
                #outputs.append("data/travel_volumes/2024-" + month + "/2024-" + month + "-"+ str(i) + "_trains.json")
        
        #for input_file, output_file in zip(inputs, outputs):
        for input_file in inputs:
            # Check if input file exists
            if not os.path.exists(input_file):
                print(f"Error: Input file '{input_file}' does not exist.")
            else:
                # Process the data
                frequencies = process_trains_from_file(input_file, output_file)
                
                print(f"\nProcessing complete!")
                print(f"Input: {input_file}")
                print(f"Output: {output_file}")
                if frequencies:
                    print(f"Total terminal pairs: {len(frequencies)}")