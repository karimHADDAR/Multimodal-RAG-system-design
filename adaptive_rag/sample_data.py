"""Small built-in corpus for smoke tests and the first demo."""

from .models import BoundingBox, EvidenceUnit


def sample_evidence() -> list[EvidenceUnit]:
    return [
        EvidenceUnit("report-2023:paragraph", "report-2023", 7, "text",
                     "Product Aurora revenue rose from $12.4M in 2022 to $15.8M in 2023."),
        EvidenceUnit("report-2023:table-1", "report-2023", 7, "table",
                     "Revenue table: Aurora 12.4M to 15.8M; Beacon 8.1M to 8.9M.",
                     BoundingBox(80, 300, 840, 280), metadata={"caption": "2022 and 2023 revenue by product"}),
        EvidenceUnit("report-2023:figure-2", "report-2023", 8, "figure", "",
                     BoundingBox(100, 180, 800, 500), metadata={"caption": "Revenue growth chart: Aurora 27.4%, Beacon 9.9%"}),
    ]