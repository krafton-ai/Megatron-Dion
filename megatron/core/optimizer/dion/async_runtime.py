"""
Dion optimizer async task runtime.
"""

from typing import Generator, List

from .constants import DEFAULT_MAX_CONCURRENT_TASKS


class AsyncTask:
    """Wrapper for async generator tasks to enable concurrent execution."""
    def __init__(self, generator: Generator[None, None, None]):
        self.generator = generator
        self.completed = False

    def step(self) -> bool:
        """Execute one step of the async task. Returns True when completed."""
        try:
            next(self.generator)
            return False  # Task not completed
        except StopIteration:
            self.completed = True
            # Close generator to free frame memory
            if self.generator is not None:
                self.generator.close()
                self.generator = None
            return True  # Task completed


class AsyncRuntime:
    """Runtime for managing and executing async tasks concurrently."""
    def __init__(self, tasks: Generator[AsyncTask, None, None],
                 max_concurrent_tasks: int = DEFAULT_MAX_CONCURRENT_TASKS):
        self.tasks: List[AsyncTask] = list(tasks)
        self.max_concurrent = max_concurrent_tasks

    def run(self):
        """Execute all tasks with controlled concurrency."""
        active_tasks: List[AsyncTask] = []
        task_iter = iter(self.tasks)

        # Initialize with first batch of tasks
        for _ in range(min(self.max_concurrent, len(self.tasks))):
            try:
                task = next(task_iter)
                active_tasks.append(task)
            except StopIteration:
                break

        # Process tasks until all completed
        while active_tasks:
            completed_indices = []

            # Step through all active tasks
            for i, task in enumerate(active_tasks):
                if task.step():
                    completed_indices.append(i)

            # Remove completed tasks and add new ones
            for i in reversed(completed_indices):
                active_tasks.pop(i)
                try:
                    # Add next task if available
                    active_tasks.append(next(task_iter))
                except StopIteration:
                    pass

        # Close all generators to free memory
        for task in self.tasks:
            if task.generator is not None:
                task.generator.close()
                task.generator = None
        self.tasks.clear()
        del active_tasks, task_iter
