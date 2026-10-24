import pytest

from helpdesk.policy import redact, redact_obj


@pytest.mark.parametrize(
    "text, label",
    [
        ("contact me at jane.doe@example.com", "EMAIL"),
        ("call +1 415-555-0134 please", "PHONE"),
        ("my number is (415) 555-0134", "PHONE"),
        ("card 4111 1111 1111 1111 was charged", "CARD"),
        ("ssn 123-45-6789", "SSN"),
        ("login from 192.168.10.4", "IP"),
        ("key sk-proj-abcdefghijklmnop1234", "API_KEY"),
    ],
)
def test_redacts_pii(text, label):
    out = redact(text)
    assert f"[REDACTED_{label}]" in out


def test_redacts_inline_password():
    out = redact("my password: Hunter2! does not work")
    assert "Hunter2" not in out
    assert "password: [REDACTED_SECRET]" in out


@pytest.mark.parametrize(
    "text",
    [
        "I cannot log in; the system says my password is incorrect.",
        "Customer CUST_05883 on ticket TCKT_025902",
        "Refund of 49.99 for invoice 2023",
    ],
)
def test_leaves_non_pii_alone(text):
    assert redact(text) == text


def test_redact_obj_recurses_and_masks_sensitive_keys():
    obj = {
        "query": "email me at a@b.io",
        "password": "plaintext",
        "nested": [{"note": "ip 10.0.0.1"}, 42],
    }
    out = redact_obj(obj)
    assert out["query"] == "email me at [REDACTED_EMAIL]"
    assert out["password"] == "[REDACTED_SECRET]"
    assert out["nested"][0]["note"] == "ip [REDACTED_IP]"
    assert out["nested"][1] == 42
    assert obj["password"] == "plaintext"  # input not mutated
