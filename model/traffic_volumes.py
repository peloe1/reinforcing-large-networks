import json
from collections import defaultdict
import os

# Updated parameters with your real stations
SUBNETWORK_STATIONS = {
    "KUO": "Kuopio",           # Internal station
    "SOR": "Sorsasalo", 
    "TOI": "Toivala",
    "SIJ": "Siilinjärvi",
    "APT": "Alapitkä",
    "LNA": "Lapinlahti",       # Internal station  
    "SKM": "Sänkimäki",        # Eastern branch
    "KNH": "Kinahmi",          # Eastern branch  
    "JKI": "Juankoski",        # Internal station
    "KRM": "Kärämä",           # Southern terminal area
    "TE": "Tervo",             # Northern area
    "OHM": "Ohenmäki",         # Northern terminal
    "LUI": "Luikonlahti"       # Eastern terminal
}

# Define the network topology (connections between stations)
NETWORK_TOPOLOGY = {
    "KRM": ["KUO"],           # South terminal to Kuopio
    "KUO": ["KRM", "SOR"],    # South to main line
    "SOR": ["KUO", "TOI"],    # Connection point
    "TOI": ["SOR", "SIJ"],    # Main line
    "SIJ": ["TOI", "APT", "SKM"],  # Junction to eastern branch
    "APT": ["SIJ", "LNA"],    # Main line
    "LNA": ["APT", "TE"],     # To northern area        
    "TE": ["LNA", "OHM"],     # To north terminal
    "OHM": ["TE"],            # North terminal
    "SKM": ["SIJ", "KNH"],    # Eastern branch
    "KNH": ["SKM", "JKI"],    # Eastern branch
    "JKI": ["KNH", "LUI"],    # To east terminal
    "LUI": ["JKI"]            # East terminal
}

# Terminal nodes
TERMINALS = {
    "south": "KRM",
    "north": "OHM", 
    "east": "LUI"
}

# All stations that should be tracked (subnetwork + terminals)
ALL_TRACKED_STATIONS = set(SUBNETWORK_STATIONS.keys()) | set(TERMINALS.values())

def analyze_traffic_frequencies(train_data):
    """
    Analyze traffic frequencies for all possible terminal pairs and internal paths.
    """
    
    frequencies = defaultdict(int)
    
    # Get all timetable rows for this train
    rows = train_data.get('timeTableRows', [])
    
    # Filter to only include tracked stations (subnetwork + terminals)
    tracked_stops = []
    for row in rows:
        station = row.get('stationShortCode', '')
        if station in ALL_TRACKED_STATIONS:
            tracked_stops.append({
                'station': station,
                'type': row.get('type'),
                'time': row.get('actualTime') or row.get('scheduledTime'),
                'trainStopping': row.get('trainStopping', False)
            })
    
    # Sort by time to get journey order
    tracked_stops.sort(key=lambda x: x['time'])
    
    # Extract the sequence of stations visited (only arrivals)
    station_sequence = []
    for stop in tracked_stops:
        if stop['type'] == 'ARRIVAL':
            station_sequence.append(stop['station'])
    
    if len(station_sequence) < 2:
        return frequencies
    
    # Determine entry and exit terminals
    entry_station = station_sequence[0]
    exit_station = station_sequence[-1]
    
    # Map stations to terminals
    entry_terminal = None
    exit_terminal = None
    
    for terminal, station in TERMINALS.items():
        if entry_station == station:
            entry_terminal = terminal
        if exit_station == station:
            exit_terminal = terminal
    
    # Also check if entry/exit stations are internal but we should map to terminals
    # For example, if train starts at KUO but KRM is the terminal, we still track it
    if not entry_terminal and entry_station in SUBNETWORK_STATIONS:
        # Check if this is actually a terminal entry point
        if entry_station == "KUO":  # Main southern entry point
            entry_terminal = "south"
        elif entry_station == "OHM":  # Main northern entry point  
            entry_terminal = "north"
        elif entry_station == "LUI":  # Main eastern entry point
            entry_terminal = "east"
    
    if not exit_terminal and exit_station in SUBNETWORK_STATIONS:
        # Check if this is actually a terminal exit point
        if exit_station == "KUO":  # Main southern exit point
            exit_terminal = "south"
        elif exit_station == "OHM":  # Main northern exit point  
            exit_terminal = "north"
        elif exit_station == "LUI":  # Main eastern exit point
            exit_terminal = "east"
    
    # If both entry and exit are terminals, it's a through journey
    if entry_terminal and exit_terminal:
        frequencies[(entry_terminal, exit_terminal)] += 1
        print(f"Train {train_data.get('trainNumber')}: {entry_terminal} → {exit_terminal} (terminal to terminal)")
    
    # Also track internal station pairs for journeys that start/end within network
    else:
        # Journey starts at a terminal but ends internally
        if entry_terminal:
            internal_exit = exit_station
            frequencies[(entry_terminal, internal_exit)] += 1
            print(f"Train {train_data.get('trainNumber')}: {entry_terminal} → {internal_exit} (terminal to internal)")
        
        # Journey ends at a terminal but starts internally  
        elif exit_terminal:
            internal_start = entry_station
            frequencies[(internal_start, exit_terminal)] += 1
            print(f"Train {train_data.get('trainNumber')}: {internal_start} → {exit_terminal} (internal to terminal)")
        
        # Completely internal journey (starts and ends within network)
        else:
            frequencies[(entry_station, exit_station)] += 1
            print(f"Train {train_data.get('trainNumber')}: {entry_station} → {exit_station} (internal journey)")
    
    # Also track all consecutive station pairs for path analysis
    for i in range(len(station_sequence) - 1):
        from_station = station_sequence[i]
        to_station = station_sequence[i + 1]
        frequencies[(from_station, to_station)] += 1
    
    return frequencies

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
                # Convert string like "('south', 'north')" back to tuple
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
        key_str = str(key)  # Convert tuple to string like "('south', 'north')"
        output_dict[key_str] = value
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_dict, f, indent=2, ensure_ascii=False)
    
    print(f"Frequencies saved to {output_file}")

def print_detailed_summary(frequencies):
    """
    Print a detailed summary organized by journey type.
    """
    if not frequencies:
        print("No frequency data to display.")
        return
        
    print("\n=== DETAILED FREQUENCY SUMMARY ===")
    
    # Group frequencies by type
    terminal_to_terminal = {}
    terminal_to_internal = {}
    internal_to_terminal = {}
    internal_journeys = {}
    station_pairs = {}
    
    for (from_node, to_node), freq in frequencies.items():
        from_is_terminal = from_node in TERMINALS
        to_is_terminal = to_node in TERMINALS
        
        if from_is_terminal and to_is_terminal:
            terminal_to_terminal[(from_node, to_node)] = freq
        elif from_is_terminal and not to_is_terminal:
            terminal_to_internal[(from_node, to_node)] = freq
        elif not from_is_terminal and to_is_terminal:
            internal_to_terminal[(from_node, to_node)] = freq
        elif not from_is_terminal and not to_is_terminal:
            # Check if it's a station pair or internal journey
            if from_node in ALL_TRACKED_STATIONS and to_node in ALL_TRACKED_STATIONS:
                station_pairs[(from_node, to_node)] = freq
            else:
                internal_journeys[(from_node, to_node)] = freq
    
    # Print summaries
    if terminal_to_terminal:
        print("\nTERMINAL-TO-TERMINAL JOURNEYS:")
        print("-" * 35)
        for (from_t, to_t), freq in sorted(terminal_to_terminal.items()):
            print(f"  {from_t} → {to_t}: {freq} train(s)")
    
    if terminal_to_internal:
        print("\nTERMINAL-TO-INTERNAL JOURNEYS:")
        print("-" * 35)
        for (from_t, to_s), freq in sorted(terminal_to_internal.items()):
            to_name = SUBNETWORK_STATIONS.get(to_s, to_s)
            print(f"  {from_t} → {to_s} ({to_name}): {freq} train(s)")
    
    if internal_to_terminal:
        print("\nINTERNAL-TO-TERMINAL JOURNEYS:")
        print("-" * 35)
        for (from_s, to_t), freq in sorted(internal_to_terminal.items()):
            from_name = SUBNETWORK_STATIONS.get(from_s, from_s)
            print(f"  {from_s} ({from_name}) → {to_t}: {freq} train(s)")
    
    if internal_journeys:
        print("\nINTERNAL JOURNEYS:")
        print("-" * 35)
        for (from_s, to_s), freq in sorted(internal_journeys.items()):
            from_name = SUBNETWORK_STATIONS.get(from_s, from_s)
            to_name = SUBNETWORK_STATIONS.get(to_s, to_s)
            print(f"  {from_s} ({from_name}) → {to_s} ({to_name}): {freq} train(s)")
    
    if station_pairs:
        print("\nCONSECUTIVE STATION PAIRS:")
        print("-" * 35)
        for (from_s, to_s), freq in sorted(station_pairs.items()):
            from_name = SUBNETWORK_STATIONS.get(from_s, from_s)
            to_name = SUBNETWORK_STATIONS.get(to_s, to_s)
            print(f"  {from_s} → {to_s}: {freq} train(s)")

def process_trains_from_file(input_file, output_file):
    """
    Main function to process trains from JSON file and update frequencies.
    """
    # Load existing frequencies
    all_frequencies = load_existing_frequencies(output_file)
    
    # Load train data
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            trains_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        print("Please check the file path and try again.")
        return all_frequencies
    except Exception as e:
        print(f"Error loading train data: {e}")
        return all_frequencies
    
    # Handle both single train and list of trains
    if isinstance(trains_data, dict):
        trains_data = [trains_data]  # Convert single train to list
    elif not isinstance(trains_data, list):
        print("Invalid data format: expected dict or list of dicts")
        return all_frequencies
    
    print(f"Processing {len(trains_data)} trains...")
    
    # Process each train
    processed_count = 0
    for train in trains_data:
        if isinstance(train, dict):
            frequencies = analyze_traffic_frequencies(train)
            
            # Increment the frequencies
            for terminal_pair, count in frequencies.items():
                all_frequencies[terminal_pair] += count
            processed_count += 1
    
    print(f"Successfully processed {processed_count} trains")
    
    # Print detailed summary
    print_detailed_summary(all_frequencies)
    
    # Save updated frequencies
    save_frequencies(all_frequencies, output_file)
    
    return all_frequencies

import json
from collections import defaultdict
import os
import glob

# ... (keep all the station definitions and functions the same as before)

def find_train_data_files():
    """
    Try to find train data files in common locations.
    """
    possible_locations = [
        "data/juna_data/*.json",
        "juna_data/*.json", 
        "*.json",
        "../data/juna_data/*.json",
        "../*.json"
    ]
    
    found_files = []
    for location in possible_locations:
        files = glob.glob(location)
        for file in files:
            if "train" in file.lower() or "juna" in file.lower() or "data" in file.lower():
                found_files.append(file)
    
    return found_files

import sys

if __name__ == "__main__":
    input_file = "data/juna_data/2024-01-01_trains.json"
    output_file = "data/travel_volumes/2024-01-01_trains.json"
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' does not exist.")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Looking for file at: {os.path.abspath(input_file)}")
        print("\nPlease make sure:")
        print("1. Your train data file exists at that location")
        print("2. The file structure is: project_folder/data/juna_data/2024-01-01_trains.json")
        print("3. Your Python file is in: project_folder/some_folder/traffic_volumes.py")
        
        # Create empty frequencies file
        empty_frequencies = defaultdict(int)
        save_frequencies(empty_frequencies, output_file)
        print(f"\nCreated empty frequencies file at {output_file}")
    else:
        # Process the data
        frequencies = process_trains_from_file(input_file, output_file)
        
        print(f"\nProcessing complete!")
        print(f"Input: {input_file}")
        print(f"Output: {output_file}")
        if frequencies:
            print(f"Total unique paths: {len(frequencies)}")