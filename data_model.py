from pydantic import BaseModel, ValidationError, validator
from typing import Optional
from typing_extensions import Literal
#constants
VALUE_1 = "1-9"
VALUE_2 = "10-99"
VALUE_3 = "99+"
VALUE_4 = "unknown"

class Company(BaseModel):
    name: str
    employees: Optional[Literal[VALUE_1, VALUE_2, VALUE_3, VALUE_4]] = VALUE_4

    @validator('employees', pre=True)
    @classmethod
    def employees_validation(cls, value) -> str:
        if value in {VALUE_1, VALUE_2, VALUE_3, VALUE_4}:
            return value
        if type(value) is str:
            value = int(''.join([char for char in value if char.isdigit()]))
        elif type(value) is not int:
            try:
                value = int(value)
            except:
                raise ValidationError
        if value < 1:
            raise ValidationError
        return VALUE_1 if value < 10 else VALUE_2 if value < 100 else VALUE_3


def test():
    test_data_list = [
        {'name': 'Random company A', 'employees': '1'},
        {'name': 'Random company B', 'employees': '67'},
        {'name': 'Random company C', 'employees': '101'},
        {'name': 'Random company D', 'employees': ' 878'},  # the whitespace in the stringed number is deliberate !
        {'name': 'Random company E', 'employees': '0'},
        {'name': 'Random company F', 'employees': 6},
    ]

    for test_data in test_data_list:
        try:
            company = Company(**test_data)
            print(f"{company.name} has {company.employees} number of employees")
        except ValidationError:
            print(f"Invalid data supplied")


if __name__ == '__main__':
    data = {'name': 'Good Company B.V.', 'employees': '1-9'}
    try:
        company = Company(**data)
        print(f"{company.name} has {company.employees} number of employees")
    except ValidationError:
        print(f"Invalid data supplied")
        raise
    test()
