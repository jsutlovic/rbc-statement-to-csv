from __future__ import annotations
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from functools import partial
from io import StringIO
from itertools import count, takewhile
import logging
from pathlib import Path
import re
import sys
import typing
from typing import (
    Dict,
    Iterator,
    List,
    NamedTuple,
    Optional,
    Pattern,
    Sequence,
    Tuple,
    Union,
    cast,
    get_args,
    get_origin,
)

import click
from dateutil.parser import parse as date_parse
from parsel import Selector
from pdfminer.high_level import extract_text_to_fp
from pdfminer.layout import LAParams
from tabulate import tabulate


# Generate XML file:
# pdf2txt.py --output_type xml --outfile - -A -L 0.51 -F +0.8 -V test_may_2023.pdf | xmllint --format - > test_may_2023_new.xml


log = logging.getLogger("convert_chequing")


class ParsingError(click.ClickException):
    """Raised if there was an issue processing the PDF as XML."""

    pass


class ProcessingException(click.ClickException):
    """Raised if there was an issue while converting XML to transactions.

    Usually this is related to malformed/unexpected data."""

    exit_code = 2


class ValidationException(click.ClickException):
    """Raised if validation of a PDFs transactions failed."""

    exit_code = 3

    def __str__(self) -> str:
        return "Validation failed: " + super().__str__()


class BBox(NamedTuple):
    left: float
    bottom: float
    right: float
    top: float

    def __str__(self) -> str:
        return f"(l: {self.left: 8.3f}, b: {self.bottom: 8.3f}, r: {self.right: 8.3f}, t: {self.top: 8.3f})"

    @staticmethod
    def from_tag(tag: Selector) -> BBox:
        # Expect a comma-separated list of 4 coordinates in order: left, top, right, bottom
        # Parse them into a list of strings
        str_coords = tag.attrib.get("bbox", "0,0,0,0").split(",")
        # Use map to convert to a list of floats
        coords = map(float, str_coords)
        # Unpack coords
        return BBox(*coords)

    @staticmethod
    def get_bounding(first: BBox, last: BBox) -> BBox:
        return BBox(first.left, first.bottom, last.right, last.top)


class TextLine(NamedTuple):
    bbox: BBox
    parent_id: int
    text: str
    font: str
    size: float

    def __str__(self) -> str:
        return f"Textline: {self.bbox}; {self.font: >24}@{self.size:<4} {self.text!r}"

    @staticmethod
    def from_tag(tag: Selector) -> TextLine:
        bbox = BBox.from_tag(tag)
        text = "".join(tag.xpath("./text/text()").getall()).strip()
        parent_id = int(tag.xpath("parent::textbox/@id").extract_first(default="0"))
        font = tag.xpath("text/@font").extract_first(default="Unknown")
        size = float(tag.xpath("text/@size").extract_first(default="0"))
        return TextLine(bbox, parent_id, text, font, size)


class LinePart(NamedTuple):
    bbox: BBox

    @staticmethod
    def from_tag(linepart: Selector) -> LinePart:
        bbox = BBox.from_tag(linepart)
        return LinePart(bbox)


class Separator(NamedTuple):
    bbox: BBox

    @staticmethod
    def from_lineparts(lineparts: List[LinePart]) -> Separator:
        bbox = BBox.get_bounding(lineparts[0].bbox, lineparts[-1].bbox)
        return Separator(bbox)


class Transaction(NamedTuple):
    date: date
    description: str = ""
    withdrawals: Optional[Decimal] = None
    deposits: Optional[Decimal] = None
    balance: Optional[Decimal] = None


# class OutputType(Enum):
#     pass

Component = Union[TextLine, Separator]
Components = Sequence[Component]


def check_type(t: type, tt: type) -> bool:
    if get_origin(tt) is Union:
        if t in get_args(tt):
            return True
    if t == tt:
        return True

    return False


class PageProcessingSettings(NamedTuple):
    """Settings object for PageComponents processing.

    `textline_xpath` is an XPath string that extracts all <textline> tags that
    are considered valid for the document and transactions.

    `details_sentinel` is a string signaling any `Component` found after it is
    valid transaction data.

    `date_matcher` is a :class:`re.Pattern` with two groups that fully matches a
    :class:`TextLine` value with the starting and ending date of the PDF statement.

    `page_number_matcher` is a :class:`re.Pattern` that matches on text like
    "1 of 2" to eliminate page numbers from being associated with transactions.

    `decimal_replace_matcher` is a :class:`re.Pattern` that is used to remove
    invalid characters for the purpose of converting numeric transaction values
    to :class:`Decimal`s. It will convert a string like "$1,234.56" to "1234.56".
    """

    textline_xpath: str = (
        ".//textbox[not(@wmode = 'vertical')]/textline[text/@size >= 8]"
    )
    details_sentinel: str = "Details of your account"
    date_matcher: Pattern = re.compile(r"^From (.+) to (.+)$")
    page_number_matcher: Pattern = re.compile(r"^(\d+) of (\d+)$")
    decimal_replace_matcher: Pattern = re.compile(r"[$,]")


class PageComponents(NamedTuple):
    """Wrapper for processing a PDF page into :class:`Component`s and :class:`Transaction`s."""

    start_date: date
    end_date: date
    components: Components
    settings: PageProcessingSettings

    @staticmethod
    def from_tag(
        settings: PageProcessingSettings, page_tag: Selector
    ) -> PageComponents:
        """Takes XML wrapped in a Selector and attempts to extract transaction
        components (text and separators) for a page.

        Starts by combining all lines of text (textline tags produced by pdfminer.six)
        into TextLine objects, excluding detected vertical text, and looks for a
        sentinel value like "Details of your account", as well as starting and ending
        dates. Ignores page numbers by regex. Next, it combines visual line segments
        (line tags) into Separator components. With all components processed, it sorts
        them by vertical and horizontal position, then returns everything after the
        sentinel TextLine.

        Returns PageComponents with the resulting Components and parsed dates.
        """

        page_id = page_tag.attrib.get("id", "0")
        log.info(f"Processing page {page_id}")

        start_date: Optional[date] = None
        end_date: Optional[date] = None
        page_transaction_components: Components = []
        sentinel_found = None

        # Find all textboxes that are not vertical and are larger than size 8
        for textline_tag in page_tag.xpath(settings.textline_xpath):
            textline = TextLine.from_tag(textline_tag)
            log.info(f"Got {textline}")
            page_transaction_components.append(textline)

            if textline.text.startswith(settings.details_sentinel):
                log.info(f"sentinel found: {textline.text!r}")
                sentinel_found = textline

            date_matches = settings.date_matcher.match(textline.text)
            if date_matches:
                log.info(f"found page dates: {textline}")
                try:
                    start_date_str, end_date_str = date_matches.groups()
                    start_date = date_parse(start_date_str).date()
                    end_date = date_parse(end_date_str).date()
                except Exception as exc:
                    log.warning(f"Could not parse dates: {exc}")
            elif settings.page_number_matcher.match(textline.text):
                # Skip page numbers
                page_transaction_components.pop()

        sep_lines = []
        # Filter for all lines
        for line_tag in page_tag.xpath(".//line"):
            linepart = LinePart.from_tag(line_tag)

            # Ignore lines that are not "below" the sentinel
            # Vertical coordinates start from the bottom of the page
            if (
                sentinel_found is not None
                and linepart.bbox.bottom > sentinel_found.bbox.bottom
            ):
                continue

            first_linepart = next(iter(sep_lines), None)
            if (
                first_linepart is None
                or first_linepart.bbox.bottom == linepart.bbox.bottom
            ):
                sep_lines.append(linepart)
            else:
                sep = Separator.from_lineparts(sep_lines)
                log.info(f"Separator: {sep}")
                page_transaction_components.append(sep)
                sep_lines = [linepart]
        else:
            if sep_lines:
                sep = Separator.from_lineparts(sep_lines)
                page_transaction_components.append(sep)

        # Raise if we can't find transaction dates
        if type(start_date) is None or type(end_date) is None:
            raise ProcessingException("Invalid document: unable to find page dates")

        # Explicit typing fix for the above check
        start_date = cast(date, start_date)
        end_date = cast(date, end_date)

        # Sort the parsed transaction components vertically (inversed), then horizontally
        page_transaction_components.sort(
            key=lambda item: (-(item.bbox.top), item.bbox.left)
        )

        # Didn't find sentinel, so there probably aren't transactions here
        if sentinel_found is None:
            raise ProcessingException(
                f"Sentinel text ({settings.details_sentinel!r}) not found in page"
            )

        # Take all items after sentinel
        sentinel_index = page_transaction_components.index(sentinel_found)
        if sentinel_index == -1:
            raise ProcessingException("Sentinel text not in components")
        if len(page_transaction_components) < sentinel_index + 1:
            raise ProcessingException("No transaction components after sentinel text")

        return PageComponents(
            start_date=start_date,
            end_date=end_date,
            components=page_transaction_components[sentinel_index + 1 :],
            settings=settings,
        )

    def to_transactions(
        self: PageComponents, description_separator=" "
    ) -> Sequence[Transaction]:
        """Processes extracted Components into Transactions.

        Takes TextLine components until it encounters a Separator, aligns the
        TextLine components to column headers (processed as the first row of
        components), attempts to parse as Decimal or date types, and returns a
        list of Transactions."""
        it_components = iter(self.components)

        # First line should be headers
        header_components = take_transaction_components(it_components)

        if not validate_transaction_headers(header_components):
            raise ProcessingException(f"Invalid headers! {header_components}")

        # Map fields to header components: {"date": TextLine, "description": TextLine, ...}
        header_fields = dict(zip(Transaction._fields, header_components))
        header_mapper = partial(map_component_to_field, header_fields)
        transactions = []

        # Use datetime to avoid issues with date_parse()
        last_date = datetime.combine(self.start_date, datetime.min.time())
        tx_field_types = typing.get_type_hints(Transaction)
        while True:
            tx_components = take_transaction_components(it_components)
            if not tx_components:
                break

            tx_dict = {}
            # Collect final transaction component info
            # Final type conversion for dates and Decimal
            # TODO: this may be good to extract into a method
            for component in tx_components:
                key, comp = header_mapper(component)
                key_type = tx_field_types.get(key, str)
                log.info(f"Got key {key} with key_type {key_type} for {comp.text!r}")

                if check_type(str, key_type):
                    tx_dict[key] = tx_dict.get(key, []) + [comp.text]
                elif check_type(date, key_type):
                    try:
                        tx_date = date_parse(comp.text, default=last_date)

                        # This probably means we're in a new year
                        if tx_date < last_date:
                            log.info(
                                "Transaction date is earlier than previous transaction, assuming new year"
                            )
                            tx_date = tx_date.replace(year=self.end_date.year)

                        last_date = tx_date
                        tx_dict[key] = tx_date.date()

                    except Exception as exc:
                        log.warning(f"Bad date: {comp.text!r} ({exc})")
                elif check_type(Decimal, key_type):
                    try:
                        num_str = self.settings.decimal_replace_matcher.sub(
                            "", comp.text
                        )
                        n = Decimal(num_str)
                        tx_dict[key] = n
                    except:
                        log.warning(f"Bad decimal: {comp.text!r}")

            if tx_dict:
                log.info(f"Got transaction: {tx_dict}")
                if not "date" in tx_dict:
                    tx_dict["date"] = last_date.date()
                tx_dict["description"] = description_separator.join(
                    tx_dict["description"]
                )
                transactions.append(Transaction(**tx_dict))

        return transactions


def take_transaction_components(components: Iterator[Component]) -> Sequence[TextLine]:
    """
    Accumulate all the components considered part of a single transaction.
    A transaction is defined as all the Components until a Separator is encountered.
    """
    not_separator = lambda item: type(item) != Separator
    tx_components = []
    for item in takewhile(not_separator, components):
        tx_components.append(item)
    return tx_components


def validate_transaction_headers(header_text: Components) -> bool:
    if len(header_text) != len(Transaction._fields):
        return False

    # Ensure headers are Text types
    types_match = [type(text) is TextLine for text in header_text]
    if not all(types_match):
        return False

    header_text = cast(List[TextLine], header_text)

    # Ensure all header text fields match expected transaction fields
    fields_match = [
        header_text.text.lower().startswith(fieldname)
        for header_text, fieldname in zip(header_text, Transaction._fields)
    ]
    return all(fields_match)


def map_component_to_field(
    header_fields: Dict[str, TextLine], component: TextLine
) -> Tuple[str, TextLine]:
    field_name = ""

    log.info(f"Mapping component to header: {component}")

    def is_aligned(field: str, a: Component, b: Component, threshold: float = 10.0):
        a_field = getattr(a.bbox, field)
        b_field = getattr(b.bbox, field)
        return abs(a_field - b_field) < threshold

    for field, field_type in typing.get_type_hints(Transaction).items():
        header_component = header_fields.get(field, None)
        if not header_component:
            continue

        # Numeric fields will be right-aligned, so check if right bounds are close
        if check_type(Decimal, field_type) and is_aligned(
            "right", header_component, component, 2.0
        ):
            field_name = field
            break
        elif is_aligned("left", header_component, component, 8.0):
            field_name = field
            break

    log.info(f"Mapped to {field_name}")

    return field_name, component


def parse_xml_doc(filename: str | Path) -> List[Transaction]:
    with open(filename, "r") as fp:
        xmlraw = fp.read()

    try:
        doc = Selector(text=xmlraw, type="xml")
    except:
        raise click.ClickException(f"Could not parse {filename} as XML")

    return process_pdf_xml(doc)


def process_pdf_xml(doc: Selector) -> List[Transaction]:
    all_transactions = []

    for page in doc.xpath("./page"):
        page_settings = PageProcessingSettings()
        page_comp = PageComponents.from_tag(page_settings, page)

        # Skip pages with no components (might just be text)
        if not page_comp.components:
            continue

        page_transactions = page_comp.to_transactions()
        all_transactions += page_transactions

    return all_transactions


def tabulate_results(res: List[Transaction]):
    tx = next(iter(res), None)
    log.info(f"First transaction: {tx}")

    full_table = tabulate(
        res,
        headers=Transaction._fields,
        tablefmt="rounded_grid",
        numalign="right",
        floatfmt=".2f",
    )

    print(full_table)


def parse_pdf_doc(filename: Path) -> List[Transaction]:
    xml_data = StringIO()
    layout_params = {
        "detect_vertical": True,
        "line_overlap": 0.51,
        "all_texts": True,
        "boxes_flow": 0.8,
    }
    with filename.open(mode="rb") as fp:
        try:
            extract_text_to_fp(
                inf=fp,
                outfp=xml_data,
                output_type="xml",
                laparams=LAParams(**layout_params),
                codec="",
            )
        except Exception as e:
            raise click.ClickException(str(e))

    try:
        doc = Selector(text=xml_data.getvalue(), type="xml")
    except Exception as e:
        raise ParsingError(str(e))

    return process_pdf_xml(doc)


def validate_transactions(transactions: Sequence[Transaction]) -> bool:
    """Check if a page of transactions mathematically align.

    Starts with an opening balance
    """
    it = iter(transactions)

    try:
        opening_tx = next(it)
    except StopIteration:
        raise ValidationException("No transactions found")

    if opening_tx.balance is None:
        raise ValidationException(
            f"Opening transaction did not contain a balance ({opening_tx})"
        )

    rb = opening_tx.balance
    log.info(f"Opening balance: {rb} ({opening_tx})")

    tx = None
    for tx in it:
        w = tx.withdrawals
        d = tx.deposits
        b = tx.balance

        if w is not None and d is not None:
            raise ValidationException(
                f"Invalid transaction contains withdrawal and deposit ({tx})"
            )

        if w is not None:
            rb -= w
        if d is not None:
            rb += d

        log.info(f"New balance: {rb} ({tx})")

        if b is not None and rb != b:
            raise ValidationException(
                f"Running balance and transaction balance do not match: {rb} != {b} ({tx})"
            )

    if tx is None:
        raise ValidationException(
            "Not enough transactions: expected opening and closing balance"
        )
    elif tx.balance is None:
        raise ValidationException(
            f"Last transaction should be closing balance, balance missing ({tx})"
        )

    return True


@click.command()
@click.argument(
    "pdf_file",
    type=click.Path(exists=True, readable=True, dir_okay=False, path_type=Path),
)
@click.option("-v", "--verbose", count=True, help="Verbose output")
@click.option(
    "-q", "--quiet", is_flag=True, default=False, help="No output will appear"
)
@click.option(
    "--validate",
    is_flag=True,
    default=False,
    help="Validate transactions against balances",
)
def convert(pdf_file: Path, verbose: int, quiet: bool, validate: bool):
    """Converts RBC Chequing account statements to machine-readable formats"""

    # Logging config
    log_level = max(logging.DEBUG, logging.ERROR - (verbose * 10))
    if quiet:
        log_level = logging.ERROR + 1
    logging.basicConfig(stream=sys.stdout, level=log_level)

    transactions = parse_pdf_doc(pdf_file)

    if validate:
        validate_transactions(transactions)

    if not quiet:
        tabulate_results(transactions)


if __name__ == "__main__":
    convert()
