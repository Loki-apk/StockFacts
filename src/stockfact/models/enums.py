"""Fixed sets of labels. The LLM agents pick from these instead of making up their own wording."""

from enum import StrEnum


class Sentiment(StrEnum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class Severity(StrEnum):
    INFO = "info"
    WATCH = "watch"
    ELEVATED = "elevated"


class Theme(StrEnum):
    EARNINGS = "earnings"
    GUIDANCE = "guidance"
    PRODUCT_LAUNCH = "product_launch"
    MANAGEMENT_CHANGE = "management_change"
    MA = "m_and_a"
    LITIGATION = "litigation"
    REGULATORY = "regulatory"
    ANALYST_ACTION = "analyst_action"
    CAPITAL_RETURN = "capital_return"  # buybacks, dividends
    LABOR = "labor"
    SUPPLY_CHAIN = "supply_chain"
    MACRO = "macro"
    OTHER = "other"


class FlagType(StrEnum):
    EARNINGS_UPCOMING = "earnings_upcoming"
    INSIDER_SELLING = "insider_selling"
    INSIDER_BUYING = "insider_buying"
    HIGH_SHORT_INTEREST = "high_short_interest"
    LITIGATION = "litigation"
    ANALYST_DOWNGRADE = "analyst_downgrade"
    ANALYST_UPGRADE = "analyst_upgrade"
    DIVIDEND_CHANGE = "dividend_change"
    DEBT_MATURITY = "debt_maturity"
    LOCKUP_EXPIRY = "lockup_expiry"
    GOING_CONCERN = "going_concern"
    RESTATEMENT = "restatement"
    FILING_RISK_FACTOR = "filing_risk_factor"
