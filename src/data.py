import csv
from functools import partial
from io import StringIO
import requests


EEG_TRACKS = frozenset([
    'BIS/EEG1_WAV',
    'BIS/EEG2_WAV',
])

ECG_TRACKS = frozenset([
    'SNUADC/ECG_II',
    'SNUADC/ECG_V5',
])

ABP_TRACKS = frozenset([
    'SNUADC/ART',
    'CardioQ/ABP',
])

RELEVANT_TRACKS = EEG_TRACKS.union(ECG_TRACKS).union(ABP_TRACKS)

def url_to_reader(url_string):
    response = requests.get(url_string)
    file = StringIO(response.text)
    return csv.DictReader(file, delimiter=',')

def get_unique_vals(key, iterable):
    return set(map(lambda item: item[key], iterable))

def case_filter(case):
    if float(case['age']) < 18.0 or (float(case['aneend']) - float(case['anestart']) == 0):
        return False
    else:
        return True

def track_filter(case_id, case_dict):
    tracks_set = set(
        map(
            lambda t: t['tname'],
            case_dict[case_id]['tracks']
        )
    )
    return (
        not tracks_set.isdisjoint(EEG_TRACKS) and
        not tracks_set.isdisjoint(ECG_TRACKS) and
        not tracks_set.isdisjoint(ABP_TRACKS)
    )

candidate_cases_by_id = {}

for case in url_to_reader('https://api.vitaldb.net/cases'):
    if case_filter(case):
        case['tracks'] = []
        candidate_cases_by_id[case['\ufeffcaseid']] = case

track_reader = url_to_reader('https://api.vitaldb.net/trks')

for track in track_reader:
    case_id = track['caseid']
    if track['tname'] in RELEVANT_TRACKS:
        if candidate_cases_by_id.get(case_id):
            candidate_cases_by_id[case_id]['tracks'].append(track)

case_track_filter = partial(track_filter, case_dict=candidate_cases_by_id)

qualified_case_ids = [case_id for case_id in filter(case_track_filter, candidate_cases_by_id.keys())]

# verify no heart surgery in remaining cases
unique_operations = set(map(lambda key: candidate_cases_by_id[key]['optype'], qualified_case_ids))
print(unique_operations)

print(len(qualified_case_ids))



# ECG waveforms were pre-processed with a 1–40-Hz band-pass filter and normalized using Z-score.
# EEG waveforms were pre-processed using a 0.5–50-Hz band-pass filter.
# ABP waveforms were used without pre-processing.
