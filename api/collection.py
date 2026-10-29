"""Collection review contracts; no inferred ground truth."""
FIELDS = {
 'service':'plate 또는 die', 'capture_kind':'plate_capture 또는 die_capture; 라벨 사진은 label_capture',
 'group':'동일 시안 개정 계열 또는 동일 목형/LOT 그룹', 'label':'OK 또는 NG; 미검수는 null',
 'overlay_status':'absent / present / unknown', 'reference_id':'승인 시안/기준 형상 ID',
 'revision':'시안 또는 목형 개정번호', 'lot':'생산 LOT', 'captured_at':'촬영 시각 ISO 8601',
 'camera_id':'카메라·촬영대 ID', 'lighting':'조명 조건', 'defect_type':'실제 확인한 결함 유형',
 'reviewer':'정답 검수자', 'clean_capture_id':'표시 없는 재촬영 파일 ID (있는 경우)',
}

def template():
 return {'version':'collection-v1','instructions':FIELDS,'records':[
  {'service':service,'capture_kind':kind,'group':'','label':None,'overlay_status':'unknown',
   'reference_id':'','revision':'','lot':'','captured_at':'','camera_id':'','lighting':'',
   'defect_type':'','reviewer':'','clean_capture_id':'','file':''}
  for service,kind in [('plate','plate_capture'),('die','die_capture')]],
  'notice':'수집용 양식입니다. 현재 자동 일괄 가져오기 형식은 아닙니다. 사진 정답은 화면에서 검수하세요.'}

def exclusions(s):
 reasons=[]
 if s.get('overlay_status','unknown')!='absent':reasons.append('결함 표시 없음 검수 필요')
 expected={'plate':'plate_capture','die':'die_capture'}.get(s.get('service'))
 if expected is None or s.get('capture_kind','unknown')!=expected:reasons.append('서비스에 맞는 실촬영 대상 확인 필요')
 return reasons

def queue(rows):
 return {'version':'collection-v1','total':len(rows),'items':[
  {'id':s['id'],'file':s['name'],'group':s['group'],'source_reason':s['weak_label'],
   'service':s['service'],'overlay_status':s.get('overlay_status','unknown'),
   'capture_kind':s.get('capture_kind','unknown'),'label':s['label'],
   'reference_candidates':len(s.get('references',[])),
   'actions':exclusions(s)+(['사진별 정답 검수 필요'] if not s.get('reviewer') else [])+
    (['참조 시안/형상 확보 및 대응 확인'] if not s.get('references') else ['참조 후보 대응·개정판 검수'])}
  for s in rows]}
