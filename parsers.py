"""PECT parsers module."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
from pect import Pectp, Pects


class Parser(ABC):
    """
    Abstract base class for file parsers. Defines the interface for
    reading from and writing to files.
    """

    @abstractmethod
    def read(self, path: Path, *args, **kwargs) -> Any:
        """
        Abstract method to read data from a file at the given path.

        Args:
            path: The pathlib.Path object to the file.
            *args, **kwargs: Additional arguments specific to the parser implementation.

        Returns:
            Any: The parsed data.
        """

    @abstractmethod
    def write(self, path: Path, data: Any, *args, **kwargs) -> None:
        """
        Abstract method to write data to a file at the given path.

        Args:
            path: The pathlib.Path object to the file.
            data: The data to be written.
            *args, **kwargs: Additional arguments specific to the parser implementation.
        """


class ProblemParser(Parser):
    """
    Parser for `.tim` problem files.
    """

    def read(self, path: Path) -> Pectp:
        """
        Parses a .tim file. Does not check if file is valid.

        Args:
            path: Path to the .tim file

        Returns:
            PECT problem
        """

        with open(path, "r", encoding="utf") as file:
            first_line = file.readline().split(" ")
            t = 45
            n = int(first_line[0])
            r = int(first_line[1])
            f = int(first_line[2])
            s = int(first_line[3])
            room_sizes = [int(file.readline()) for _ in range(r)]
            attends = [[int(file.readline()) for _ in range(n)] for _ in range(s)]
            roomfeatures = [[int(file.readline()) for _ in range(f)] for _ in range(r)]
            eventfeatures = [[int(file.readline()) for _ in range(f)] for _ in range(n)]
            event_availability = [
                [int(file.readline()) for _ in range(t)] for _ in range(n)
            ]
            before = [[int(file.readline()) for _ in range(n)] for _ in range(n)]

            return (
                n,
                r,
                f,
                s,
                room_sizes,
                attends,
                roomfeatures,
                eventfeatures,
                event_availability,
                before,
            )

    def write(self, path: Path, data: Pectp) -> None:
        """
        Writes PECT problem to a .tim file.

        Args:
            path: Path to the output .tim file.
            data: PECT problem
        """
        (
            n,
            r,
            f,
            s,
            room_sizes,
            attends,
            roomfeatures,
            eventfeatures,
            event_availability,
            before,
        ) = data
        with open(path, "w", encoding="utf-8") as file:
            file.write(f"{n} {r} {f} {s}\n")
            for size in room_sizes:
                file.write(f"{size}\n")
            for row in attends:
                file.write(" ".join(map(str, row)) + "\n")
            for row in roomfeatures:
                file.write(" ".join(map(str, row)) + "\n")
            for row in eventfeatures:
                file.write(" ".join(map(str, row)) + "\n")
            for row in event_availability:
                file.write(" ".join(map(str, row)) + "\n")
            for row in before:
                file.write(" ".join(map(str, row)) + "\n")


class SolutionParser(Parser):
    """
    Parser for .sln solution files.
    """

    def read(self, path: Path, pect: Pectp) -> Pects:
        """
        Reads solution from a .sln file

        Args:
            path: Path to the .sln file.
            pect: PECT problem

        Returns:
            PECT solution
        """

        n = pect[0]
        solution = []
        with open(path, "r", encoding="utf-8") as file:
            for _ in range(n):
                line = file.readline().split(" ")
                solution.append([int(line[0]), int(line[1])])
        return solution

    def write(self, path: Path, data: Pects) -> None:
        """
        Writes a solution to a .sln file

        Args:
            path: Path to the .sln file
            data: PECT solution
        """

        with open(path, "w", encoding="utf-8") as file:
            for timeslot, room in data:
                file.write(f"{timeslot} {room}\n")

    def write_formatted_solution(
        self, path: Path, pect: Pectp, solution: Pects
    ) -> None:
        """
        Writes a formatted solution to file

        Args:
            path: Path to the .formatted.sln file
            pect: PECT problem
            solution: PECT solution
        """

        (
            _,
            r,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
        ) = pect

        timetable = [[["-" for _ in range(5)] for _ in range(9)] for _ in range(r)]

        for i, event in enumerate(solution):
            timeslot, room = event
            slot = timeslot % 9
            day = timeslot // 9
            timetable[room][slot][day] = str(i)

        with open(path, "w", encoding="utf-8") as file:
            for room in range(r):
                file.write(f"ucionica{room}\n")
                file.write("        PON     UTO     SRI     CET     PET\n")
                for slot in range(9):
                    file.write("0       ")
                    for day in range(5):
                        file.write(timetable[room][slot][day].ljust(8))
                    file.write("\n")


class EvaluationParser(Parser):
    """
    Parser for evaluation files.
    """

    def read(self, path: Path) -> tuple[str, int, int, int]:
        """
        Reads a formatted evaluation from file

        Args:
            path: Path to the .eval file
        Returns:
            A tuple representing (instance, distance to feasibility, soft cost, total cost)
        """

        with open(path, "r", encoding="utf-8") as file:
            lines = file.readlines()
            instance = lines[0].strip().split(": ")[1]
            distance = int(lines[1].strip().split(": ")[1])
            soft = int(lines[2].strip().split(": ")[1])
            total = int(lines[3].strip().split(": ")[1])
            return (instance, distance, soft, total)

    def write(self, path: Path, data: tuple[int, int]) -> None:
        """
        Writes a formatted evaluation to file

        Args:
            path: string to file path
            data: tuple representing distance to feasibility and soft cost
        """

        with open(path, "w", encoding="utf-8") as file:
            file.write(f"Instance: {path.stem}\n")
            file.write(f"Distance to feasibility: {data[0]}\n")
            file.write(f"Soft cost: {data[1]}\n")
            file.write(f"Total: {data[0] + data[1]}\n")


class EvaluationCsvGenerator:
    """
    Utility class to generate a single CSV file from multiple '.eval' files.
    """

    def generate_evaluation_csv(self, path: Path) -> None:
        """
        Reads evaluation files and generates evaluation csv

        Args:
            path: Path to evaluations directory
        """

        eval_files = list(path.glob("*.eval"))
        evals = ["instance,distance_to_feasibility,soft_cost,total_cost"]

        for eval_file in eval_files:
            with open(eval_file, "r", encoding="utf-8") as file:
                lines = file.readlines()
                instance = lines[0].strip().split(": ")[1]
                distance = lines[1].strip().split(": ")[1]
                soft = lines[2].strip().split(": ")[1]
                total = lines[3].strip().split(": ")[1]
                evals.append(f"{instance},{distance},{soft},{total}")

        with open(path / "evaluation.csv", "w", encoding="utf-8") as file:
            file.write("\n".join(evals) + "\n")
