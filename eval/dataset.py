from dataclasses import dataclass
from typing import Literal

EvalCategory = Literal[
    "accurate",
    "numeric_hallucination",
    "entity_swap",
    "contradiction",
    "mixed_borderline",
]


@dataclass(frozen=True)
class EvalCase:
    case_id: str
    category: EvalCategory
    expected_action: Literal["PASS", "FLAG", "BLOCK"]
    expect_contradiction: bool
    ground_truth: str
    input_text: str


EVAL_CASES: list[EvalCase] = [
    # --- 1. Accurate Grounded Summaries (Expected: PASS) ---
    EvalCase(
        case_id="ACC-01",
        category="accurate",
        expected_action="PASS",
        expect_contradiction=False,
        ground_truth=(
            "[Page 1]\n"
            "Helios Energy reported FY2025 revenue of $420 million, up 18% from FY2024. "
            "The company operates 14 solar farms across Rajasthan and employs 1,250 full-time staff."
        ),
        input_text=(
            "Helios Energy reported FY2025 revenue of $420 million, representing an 18% increase "
            "from FY2024. The company operates 14 solar farms across Rajasthan."
        ),
    ),
    EvalCase(
        case_id="ACC-02",
        category="accurate",
        expected_action="PASS",
        expect_contradiction=False,
        ground_truth=(
            "[Page 1]\n"
            "Bharat FinTech launched its UPI Lite wallet in March 2025 under CEO Ananya Kulkarni. "
            "Within six months, the platform processed 8.4 million transactions with 99.95% uptime."
        ),
        input_text=(
            "Bharat FinTech launched its UPI Lite wallet in March 2025 under CEO Ananya Kulkarni "
            "and processed 8.4 million transactions within six months."
        ),
    ),
    EvalCase(
        case_id="ACC-03",
        category="accurate",
        expected_action="PASS",
        expect_contradiction=False,
        ground_truth=(
            "[Page 1]\n"
            "MedNova Labs completed Phase 2 clinical trials for its DengueX vaccine in Pune "
            "in October 2024 with 600 adult volunteers and reported an efficacy rate of 81.2%."
        ),
        input_text=(
            "MedNova Labs completed Phase 2 clinical trials for its DengueX vaccine in Pune "
            "in October 2024 with 600 adult volunteers, achieving an efficacy rate of 81.2%."
        ),
    ),

    # --- 2. Numeric Hallucinations (Expected: BLOCK) ---
    EvalCase(
        case_id="NUM-01",
        category="numeric_hallucination",
        expected_action="BLOCK",
        expect_contradiction=False,
        ground_truth=(
            "[Page 1]\n"
            "Acme Solar reported FY2025 revenue of $348.5 million and reduced manufacturing "
            "costs by 11.4% compared to FY2024."
        ),
        input_text=(
            "Acme Solar reported FY2025 revenue of $384.5 million and reduced manufacturing "
            "costs by 41.1% compared to FY2024."
        ),
    ),
    EvalCase(
        case_id="NUM-02",
        category="numeric_hallucination",
        expected_action="BLOCK",
        expect_contradiction=False,
        ground_truth=(
            "[Page 1]\n"
            "Indus Logistics expanded its electric fleet to 320 delivery vans in Bengaluru "
            "in 2025, cutting annual fuel expenditure by Rs 4.8 crore."
        ),
        input_text=(
            "Indus Logistics expanded its electric fleet to 950 delivery vans in Bengaluru "
            "in 2023, cutting annual fuel expenditure by Rs 18.5 crore."
        ),
    ),
    EvalCase(
        case_id="NUM-03",
        category="numeric_hallucination",
        expected_action="BLOCK",
        expect_contradiction=False,
        ground_truth=(
            "[Page 1]\n"
            "Reserve Bank guidelines issued in June 2025 set the minimum capital adequacy "
            "ratio for small finance banks at 15% and capped unsecured micro-loans at 20%."
        ),
        input_text=(
            "Reserve Bank guidelines issued in June 2025 set the minimum capital adequacy "
            "ratio for small finance banks at 9% and capped unsecured micro-loans at 50%."
        ),
    ),

    # --- 3. Entity / Name Swaps (Expected: BLOCK) ---
    EvalCase(
        case_id="ENT-01",
        category="entity_swap",
        expected_action="BLOCK",
        expect_contradiction=False,
        ground_truth=(
            "[Page 1]\n"
            "Vayu Robotics, headquartered in Hyderabad, appointed Vikram Deshmukh as Chief "
            "Technology Officer after securing Series B funding from Peak XV Partners."
        ),
        input_text=(
            "Vayu Robotics, headquartered in Chennai, appointed Rohan Malhotra as Chief "
            "Technology Officer after securing Series B funding from Sequoia China."
        ),
    ),
    EvalCase(
        case_id="ENT-02",
        category="entity_swap",
        expected_action="BLOCK",
        expect_contradiction=False,
        ground_truth=(
            "[Page 1]\n"
            "Kaveri Semiconductors signed a five-year wafer supply agreement with Tata Electronics "
            "to assemble automotive chips at its Dholera plant in Gujarat."
        ),
        input_text=(
            "Kaveri Semiconductors signed a five-year wafer supply agreement with Reliance Jio "
            "to assemble automotive chips at its Noida plant in Uttar Pradesh."
        ),
    ),
    EvalCase(
        case_id="ENT-03",
        category="entity_swap",
        expected_action="BLOCK",
        expect_contradiction=False,
        ground_truth=(
            "[Page 1]\n"
            "Dr. Kavita Iyer from IIT Madras led the open-source Tamil speech recognition "
            "project funded by the Ministry of Electronics and Information Technology."
        ),
        input_text=(
            "Dr. Suresh Nair from IIT Delhi led the open-source Bengali speech recognition "
            "project funded by the Ministry of Education."
        ),
    ),

    # --- 4. Internal Contradictions (Expected: BLOCK) ---
    EvalCase(
        case_id="CON-01",
        category="contradiction",
        expected_action="BLOCK",
        expect_contradiction=True,
        ground_truth=(
            "[Page 1]\n"
            "NovaCloud generated $120 million in annual recurring revenue in 2025 with zero "
            "layoffs during the fiscal year."
        ),
        input_text=(
            "NovaCloud generated $120 million in annual recurring revenue in 2025. However, "
            "NovaCloud's total annual recurring revenue in 2025 was only $45 million, and "
            "the company laid off 400 employees."
        ),
    ),
    EvalCase(
        case_id="CON-02",
        category="contradiction",
        expected_action="BLOCK",
        expect_contradiction=True,
        ground_truth=(
            "[Page 1]\n"
            "GreenGrid Storage opened its first battery recycling plant in Pune in January 2025 "
            "with a capacity of 5,000 tonnes per year."
        ),
        input_text=(
            "GreenGrid Storage opened its first battery recycling plant in Pune in January 2025. "
            "Later, the report states GreenGrid Storage has never opened a plant in Pune and "
            "delayed construction until 2028."
        ),
    ),
    EvalCase(
        case_id="CON-03",
        category="contradiction",
        expected_action="BLOCK",
        expect_contradiction=True,
        ground_truth=(
            "[Page 1]\n"
            "Orion Airways recorded a net profit of $64 million in Q4 2025 and operates 28 "
            "narrow-body aircraft."
        ),
        input_text=(
            "Orion Airways recorded a net profit of $64 million in Q4 2025. At the same time, "
            "Orion Airways posted a net loss of $90 million in Q4 2025 while operating 110 aircraft."
        ),
    ),

    # --- 5. Mixed / Borderline Cases: 2 True + 1 False Claim (Expected: FLAG) ---
    EvalCase(
        case_id="MIX-01",
        category="mixed_borderline",
        expected_action="FLAG",
        expect_contradiction=False,
        ground_truth=(
            "[Page 1]\n"
            "Deccan Motors sold 52,000 electric scooters in FY2025 across 18 Indian states. "
            "Its flagship scooter offers a certified range of 145 km on a single charge."
        ),
        input_text=(
            "Deccan Motors sold 52,000 electric scooters in FY2025. "
            "The company operates across 18 Indian states. "
            "Its flagship scooter offers a certified range of 310 km on a single charge."
        ),
    ),
    EvalCase(
        case_id="MIX-02",
        category="mixed_borderline",
        expected_action="FLAG",
        expect_contradiction=False,
        ground_truth=(
            "[Page 1]\n"
            "Saffron Foods exported 12,400 metric tonnes of spices in 2025 from Kochi port. "
            "European markets accounted for 34% of its total export volume."
        ),
        input_text=(
            "Saffron Foods exported 12,400 metric tonnes of spices in 2025. "
            "Exports were shipped from Kochi port. "
            "European markets accounted for 78% of its total export volume."
        ),
    ),
    EvalCase(
        case_id="MIX-03",
        category="mixed_borderline",
        expected_action="FLAG",
        expect_contradiction=False,
        ground_truth=(
            "[Page 1]\n"
            "PulseAI raised $25 million in Series A funding in April 2025 in Bengaluru. "
            "The startup was founded by Priya Menon and currently serves 140 enterprise clients."
        ),
        input_text=(
            "PulseAI raised $25 million in Series A funding in April 2025 in Bengaluru. "
            "The startup was founded by Priya Menon. "
            "PulseAI currently serves 2,500 enterprise clients."
        ),
    ),
]