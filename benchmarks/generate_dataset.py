from __future__ import annotations

import json
from pathlib import Path


OUTPUT_PATH = (
    Path(__file__).parent
    / "data"
    / "customer_prompts.json"
)


CUSTOMERS = [
    {
        "name": "Aarav Mehta",
        "age": 28,
        "email": "aarav.mehta@example.com",
        "company": "Northstar Analytics",
        "role": "Data Analyst",
    },
    {
        "name": "Priya Sharma",
        "age": 32,
        "email": "priya.sharma@example.com",
        "company": "BluePeak Systems",
        "role": "Product Manager",
    },
    {
        "name": "Rohan Das",
        "age": 26,
        "email": "rohan.das@example.com",
        "company": "Vertex Labs",
        "role": "Software Engineer",
    },
    {
        "name": "Ananya Roy",
        "age": 30,
        "email": "ananya.roy@example.com",
        "company": "CloudBridge Technologies",
        "role": "Solutions Architect",
    },
    {
        "name": "Vikram Singh",
        "age": 35,
        "email": "vikram.singh@example.com",
        "company": "Apex Consulting",
        "role": "Sales Director",
    },
    {
        "name": "Sneha Iyer",
        "age": 27,
        "email": "sneha.iyer@example.com",
        "company": "FinEdge",
        "role": "Business Analyst",
    },
    {
        "name": "Arjun Nair",
        "age": 31,
        "email": "arjun.nair@example.com",
        "company": "GreenGrid Energy",
        "role": "Operations Manager",
    },
    {
        "name": "Meera Kapoor",
        "age": 29,
        "email": "meera.kapoor@example.com",
        "company": "PixelWorks",
        "role": "UX Designer",
    },
    {
        "name": "Karan Malhotra",
        "age": 34,
        "email": "karan.malhotra@example.com",
        "company": "SecureNet",
        "role": "Security Engineer",
    },
    {
        "name": "Ishita Sen",
        "age": 25,
        "email": "ishita.sen@example.com",
        "company": "EduSphere",
        "role": "Program Coordinator",
    },
]


TEMPLATES = {
    "crm": [
        (
            "Update the CRM record from this customer note: "
            "{name} is {age} years old and works as {role} "
            "at {company}. Their email is {email}."
        ),
        (
            "A customer profile needs to be entered into our system. "
            "The contact is {name}, age {age}, employed by {company} "
            "as a {role}. Contact email: {email}."
        ),
        (
            "Convert this CRM note into the requested customer "
            "information: {name}, {age}, {role}, {company}, {email}."
        ),
        (
            "The account manager recorded the following details. "
            "Identify the customer's name, age, email, company and role. "
            "Customer: {name}. Age: {age}. Email: {email}. "
            "Company: {company}. Position: {role}."
        ),
    ],
    "support": [
        (
            "A support ticket was opened by {name}, a {role} at "
            "{company}. The customer is {age} years old and can be "
            "reached at {email}. Extract the customer details."
        ),
        (
            "Support case metadata: requester={name}; "
            "requester_age={age}; organization={company}; "
            "job_title={role}; contact={email}. "
            "Prepare the customer record."
        ),
        (
            "Summarize the identity details from this support request. "
            "{name} ({role}, {company}) is {age}. Email: {email}."
        ),
        (
            "The following customer contacted technical support: "
            "{name}. They work at {company} as a {role}, are {age}, "
            "and use {email}. Extract the structured customer data."
        ),
    ],
    "sales": [
        (
            "Sales lead information: {name}, {age}, {role} at "
            "{company}. Email address: {email}. Create the lead record."
        ),
        (
            "A salesperson entered this lead into the notes: "
            "{name} works for {company} as a {role}. "
            "They are {age} years old and their email is {email}. "
            "Extract the relevant fields."
        ),
        (
            "Turn this prospect description into customer information: "
            "{name}, age {age}, {company}, {role}, {email}."
        ),
        (
            "New business contact: {name} | {role} | {company} | "
            "{age} | {email}. Prepare the structured record."
        ),
    ],
    "meeting": [
        (
            "Meeting notes mention {name}, who is {age} and works at "
            "{company} as a {role}. Their contact address is {email}. "
            "Extract the person's details."
        ),
        (
            "During today's meeting, {name} introduced themselves as "
            "a {role} from {company}. Their age is {age} and their "
            "email is {email}. Convert this into structured information."
        ),
        (
            "Participant details from the meeting: {name}, {role}, "
            "{company}, age {age}, email {email}. Extract the record."
        ),
        (
            "Identify the attendee information in these notes: "
            "{name} ({age}), {role} at {company}, {email}."
        ),
    ],
    "onboarding": [
        (
            "During onboarding, {name} provided their contact details. "
            "They are {age}, work as a {role} at {company}, and use "
            "{email}. Create the customer record."
        ),
        (
            "New employee/customer onboarding information: "
            "Full name: {name}; Age: {age}; Company: {company}; "
            "Role: {role}; Email: {email}. Extract the fields."
        ),
        (
            "The onboarding specialist recorded that {name} is a "
            "{role} at {company}. Their age is {age} and their email "
            "is {email}. Return the requested information."
        ),
        (
            "Process this onboarding note: {name}, {age}, {email}, "
            "{company}, {role}. Produce the customer details."
        ),
    ],
}


def build_prompt(
    template: str,
    customer: dict[str, object],
) -> str:
    return template.format(**customer)


def generate_dataset() -> list[dict[str, object]]:
    dataset: list[dict[str, object]] = []

    prompt_id = 1

    for category, templates in TEMPLATES.items():
        for index in range(40):
            customer = CUSTOMERS[
                index % len(CUSTOMERS)
            ]

            template = templates[
                index % len(templates)
            ]

            prompt = build_prompt(
                template,
                customer,
            )

            dataset.append(
                {
                    "id": f"FG-{prompt_id:03d}",
                    "category": category,
                    "prompt": prompt,
                    "expected": customer,
                }
            )

            prompt_id += 1

    return dataset


def main() -> None:
    dataset = generate_dataset()

    if len(dataset) != 200:
        raise RuntimeError(
            f"Expected 200 prompts, got {len(dataset)}."
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        json.dumps(
            dataset,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        f"Generated {len(dataset)} benchmark prompts."
    )
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()