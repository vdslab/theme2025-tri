def is_minute_highlight(minute_data):
    # play_features に含まれる特徴量のうち1つでも True であれば True
    for feature_group in ['hit_event', 'rbi_impact']:
        for v in minute_data['play_features'][feature_group].values():
            if v:
                return True
    return False
