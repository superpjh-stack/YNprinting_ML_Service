from typing import Literal
from pydantic import BaseModel, Field, model_validator

DEFECTS = {
 'plate':['missing_text','wrong_text','smear','dot_defect','misalignment','other'],
 'die':['wear','bend','break','position_error','other'],
}
class Box(BaseModel):
    defect_type: str = Field(min_length=1,max_length=40)
    x: float = Field(ge=0,lt=1,allow_inf_nan=False)
    y: float = Field(ge=0,lt=1,allow_inf_nan=False)
    width: float = Field(gt=0,le=1,allow_inf_nan=False)
    height: float = Field(gt=0,le=1,allow_inf_nan=False)
    expected_text: str = Field(default='',max_length=2000)
    observed_text: str = Field(default='',max_length=2000)
    @model_validator(mode='after')
    def bounds(self):
        if self.x+self.width>1.000001 or self.y+self.height>1.000001:
            raise ValueError('결함 좌표는 이미지 범위 안에 있어야 합니다.')
        return self

def validate(service,label,boxes,complete):
    if any(b.defect_type not in DEFECTS.get(service,[]) for b in boxes):
        raise ValueError('서비스에 맞지 않는 결함 유형입니다.')
    if label=='OK' and boxes:raise ValueError('정상 판정에는 결함 영역을 등록할 수 없습니다.')
    if label=='NG' and complete and not boxes:raise ValueError('불량 위치 검수 완료에는 결함 영역이 필요합니다.')
