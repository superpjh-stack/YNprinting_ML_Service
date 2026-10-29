import argparse,json
from pathlib import Path
from . import core,training
p=argparse.ArgumentParser();p.add_argument('action',choices=['import','clean','features','report','train']);p.add_argument('--source');p.add_argument('--service',choices=['plate','die']);args=p.parse_args()
if args.action=='import':
    if not args.source:p.error('--source 필요')
    result=core.import_folder(args.source)
elif args.action in ['clean','features']:result=core.process(args.action)
elif args.action=='train':
    if not args.service:p.error('--service 필요')
    result=training.train(args.service)
else:result=core.overview()
print(json.dumps(result,ensure_ascii=False,indent=2))
