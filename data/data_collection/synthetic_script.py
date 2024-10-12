from data.data_collection.record_script import record
import asyncio, time
from data.data_collection.synthetic_mouse_path import generate_multiple_trajectories, move_mouse_through_trajectory

async def create_synthetic_dataset(n=1):

    trajectories = generate_multiple_trajectories(n, 256, 256)

    for i, trajectory in enumerate(trajectories):

        await record('raw_data', f'record_{i}', move_mouse_through_trajectory, trajectory)
        time.sleep(3)


if __name__ == "__main__":
    asyncio.run(create_synthetic_dataset(2))