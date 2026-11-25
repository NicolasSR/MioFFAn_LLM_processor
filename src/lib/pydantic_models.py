from pydantic import BaseModel, Field


# We try to keep all Pydantic models as close as possible to the ontology classes.
# So we use the exact names for classes and properties whenever possible.

class Written_representation(BaseModel):
    """Written representation of a specific abstract concept. Micht be plain text or LaTeX/MathML."""
    representation_str: str = Field(description="String information of a written representation. Might be textual or LaTeX, but in a way that is compilable and searchable via regex.")
    representation_type: str = Field(description="String tag stating whether the written representation is 'text', 'latex' or 'mathml'.")

class Text_representation(Written_representation):
    pass

class Mathematical_representation(Written_representation):
    pass

class Mathematical_property(BaseModel):
    """Property of a mathematical entity, written in plain text or LaTeX. Can be pure mathematical properties/relationships or statements about what they represent physically."""
    property_enunciate: str = Field(description="String in plain text or LaTeX stating the enunciate of a mathematical property")
    is_confirmed: bool = Field(description="Whether the property is explicitly confirmed within the given text")

class Mathematical_entity(BaseModel):
    """Individual mathematical object, e.g. variable, function, set..."""
    representations_list: list[Written_representation] = Field(description="List of written representations of an abstract mathematical entity. Might be textual or LaTeX/MathML.")
    properties_list: list[Mathematical_property] = Field(description="List of properties of a mathematical entity, written in plain text or LaTeX. Can be pure mathematical properties/relationships or statements about what they represent physically.")
    is_standard_operator: bool = Field(description="Bool stating whether the mathematical entity is a commonly used operator in standard notation.")

class Mathematical_entites_list(BaseModel):
    """List of individual mathematical objects, e.g. variables, functions, sets..."""
    mathematical_entities_list: list[Mathematical_entity] = Field(description="List of individual mathematical objects, e.g. variables, functions, sets...")
