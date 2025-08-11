from deep_sort_realtime.deepsort_tracker import DeepSort

def create_tracker():
    return DeepSort(
        max_age=10,
        n_init=3,
        max_cosine_distance=0.4,
        nn_budget=None,
        override_track_class=None,
        embedder="mobilenet",  # Lightweight for real-time; FaceNet is used separately
        half=True
    )
