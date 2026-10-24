"""GD&T-aware planning and simulated adaptive acquisition; lengths in micrometers.

Source: github.com/ajielee1975-arch/Aggie-Invent-LANL-UI (engine.py, version 2.0).
AMSI demo changes (backward compatible, all 15 original tests pass):
  1. characteristics texture and defect added (upper bound, evaluated without datums, like flatness)
  2. run() accepts transform= (measured part to stage transform from the camera survey)
  3. run() accepts survey_adapter= (description of the real large field survey)
"""
from datetime import date, datetime, timezone
import hashlib
import json
import math
import numpy as np

VERSION = '2.0'
TYPES = {'size', 'flatness', 'position', 'profile', 'texture', 'defect'}

def validate_features(features):
    if not isinstance(features, list) or not features:
        raise ValueError('Provide a nonempty JSON array of feature objects.')
    seen = set()
    for f in features:
        for k in ['id','characteristic','geometry','surface','access','datums','modifier','tolerance_um','nominal_um','roi_mm','xyz_mm']:
            if k not in f: raise ValueError(f'Missing feature field: {k}')
        if not isinstance(f['id'],str) or not f['id'].strip() or f['id'] in seen:
            raise ValueError('Feature IDs must be nonblank and unique.')
        seen.add(f['id'])
        for k in ['tolerance_um','nominal_um','roi_mm']:
            if not isinstance(f[k],(int,float)) or not math.isfinite(f[k]): raise ValueError(f'{k} must be finite.')
        if f['tolerance_um'] <= 0 or f['roi_mm'] <= 0: raise ValueError('Tolerance and ROI must be positive.')
        if not isinstance(f['xyz_mm'],list) or len(f['xyz_mm']) != 3 or not all(isinstance(x,(int,float)) and math.isfinite(x) for x in f['xyz_mm']):
            raise ValueError('xyz_mm must contain three finite numbers.')
        for k in ['characteristic','geometry','surface','access','datums','modifier']:
            if not isinstance(f[k],str): raise ValueError(f'{k} must be text.')
    return features

def uncertainty(instrument, registration_u, environment_u):
    # All inputs below are standard uncertainties (k=1), assumed independent.
    return 2*math.sqrt(instrument['u_cal_um']**2 + instrument['u_repeat_um']**2 +
                       (instrument['resolution_um']/math.sqrt(12))**2 +
                       registration_u**2 + environment_u**2)

def eligibility(f, instruments, day, registration_u, environment_u, ratio=4):
    records = []
    for i in instruments:
        reasons = []
        if f['characteristic'] not in i['characteristics']: reasons.append('unsupported characteristic')
        if f['geometry'] not in i['geometries']: reasons.append('incompatible geometry')
        if f['surface'] not in i['surfaces']: reasons.append('incompatible surface')
        if f['access'] not in i['access']: reasons.append('inaccessible feature')
        if f['roi_mm'] > i['field_mm']: reasons.append('ROI exceeds validated field; stitching not qualified')
        if not i['enabled']: reasons.append('instrument unavailable')
        if not (date.fromisoformat(i['calibrated_on']) <= day <= date.fromisoformat(i['calibration_due'])):
            reasons.append('calibration outside validity interval')
        if not i['certificate'] or not i['reference']: reasons.append('missing calibration reference')
        u = uncertainty(i, registration_u, environment_u)
        if u > f['tolerance_um']/ratio: reasons.append('uncertainty exceeds budget')
        records.append(dict(instrument_id=i['id'], technique=i['technique'], U_um=u,
                            cycle_s=i['cycle_s'], eligible=not reasons, reason='; '.join(reasons) or 'All configured checks passed'))
    return records

def classify(f, value, u):
    if f['characteristic']=='size':
        lo=f['nominal_um']-f['tolerance_um']/2; hi=f['nominal_um']+f['tolerance_um']/2
        if value-u >= lo and value+u <= hi: return 'Conforming'
        if value+u < lo or value-u > hi: return 'Nonconforming'
    else:
        # Scalar zone error from a qualified GD&T evaluator, upper-bound conformance.
        if value+u <= f['tolerance_um']: return 'Conforming'
        if value-u > f['tolerance_um']: return 'Nonconforming'
    return 'Indeterminate'

def run(features, instruments, *, seed=42, mode='Nominal', registration_u=.02,
        environment_u=.02, localization_error_mm=.02, ratio=4, day=None, observations=None,
        transform=None, survey_adapter=None):
    validate_features(features)
    day = day or date.today()
    for v in [registration_u,environment_u,localization_error_mm]:
        if not math.isfinite(v) or v < 0: raise ValueError('Uncertainty/localization inputs must be finite and nonnegative.')
    if not math.isfinite(ratio) or ratio < 1: raise ValueError('Uncertainty ratio must be at least 1.')
    rng=np.random.default_rng(seed); events=[]; results=[]; candidates=[]
    # Fixture transform is deliberately illustrative. Handoff coordinates are calculated,
    # not fitted from scanner data. A hardware adapter must provide a measured transform.
    if transform is None:
        transform=np.eye(4); transform[:3,3]=[100,20,5]
    else:
        transform=np.asarray(transform,float)
    def event(feature,state,**data):
        payload=dict(sequence=len(events)+1,time_utc=datetime.now(timezone.utc).isoformat(),feature_id=feature,state=state,**data)
        payload['previous_hash']=events[-1]['hash'] if events else '0'*64
        payload['hash']=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
        events.append(payload)
    event('PART','LARGE_FIELD_SURVEY',adapter=survey_adapter or 'simulated structured-light survey',transform_part_to_stage=transform.tolist(),localization_error_mm=localization_error_mm)
    for f in features:
        event(f['id'],'INTERPRET_REQUIREMENT',characteristic=f['characteristic'],datums=f['datums'],modifier=f['modifier'])
        reason=None
        if f['characteristic'] not in TYPES: reason='Unsupported GD&T characteristic'
        elif f['modifier']!='RFS': reason='MMC/LMC or other modifier requires a qualified evaluator'
        elif f['characteristic'] in ['position','profile'] and not f['datums'].strip(): reason='Datum reference frame required'
        elif f['characteristic'] in ('flatness','texture','defect') and f['datums'].strip(): reason='Flatness, texture and defect depth must be evaluated without a datum reference'
        if reason:
            event(f['id'],'BLOCKED',reason=reason); results.append(dict(feature_id=f['id'],status='Blocked',reason=reason)); continue
        options=eligibility(f,instruments,day,registration_u,environment_u,ratio)
        candidates.extend(dict(feature_id=f['id'],**x) for x in options)
        available=sorted([o for o in options if o['eligible']],key=lambda x:(x['cycle_s'],x['U_um']))
        stage_xyz=(transform@np.array([*f['xyz_mm'],1]))[:3].tolist()
        event(f['id'],'LOCALIZE_ROI',part_xyz_mm=f['xyz_mm'],stage_xyz_mm=stage_xyz,roi_mm=f['roi_mm'])
        # Require entire ROI plus positioning margin inside the target field.
        available=[o for o in available if next(i['field_mm'] for i in instruments if i['id']==o['instrument_id']) >= f['roi_mm']+2*localization_error_mm]
        if not available:
            reason='No capable instrument or localization margin exceeds available field'
            event(f['id'],'BLOCKED',reason=reason); results.append(dict(feature_id=f['id'],status='Blocked',reason=reason)); continue
        previous=None; final=None
        fraction={'Nominal':.35,'Near limit':.99,'Out of tolerance':1.3}[mode]
        truth=f['nominal_um']+(fraction*f['tolerance_um']/2 if mode!='Nominal' else 0) if f['characteristic']=='size' else fraction*f['tolerance_um']
        for o in available:
            if previous and o['U_um'] >= previous['U_um']: continue
            instrument=next(i for i in instruments if i['id']==o['instrument_id'])
            event(f['id'],'REFINE' if previous else 'SELECT_TECHNIQUE',instrument_id=instrument['id'],reason='Lower uncertainty after indeterminate result' if previous else 'Fastest compatible technique within uncertainty budget')
            event(f['id'],'REGISTER_LOCAL_FRAME',instrument_id=instrument['id'],datum_frame=f['datums'] or 'local feature frame',registration_standard_u_um=registration_u,registration='SIMULATED local datum/frame re-establishment')
            if observations is None:
                value=truth+rng.normal(0,o['U_um']/8)
                if f['characteristic']!='size': value=max(0,value)
            else:
                matches=[v for v in observations if v['feature_id']==f['id'] and v['instrument_id']==instrument['id']]
                if len(matches)!=1:
                    final=dict(feature_id=f['id'],status='Awaiting measurement',reason='Provide exactly one evaluated result for selected instrument',instrument_id=instrument['id'])
                    event(f['id'],'AWAITING_MEASUREMENT',instrument_id=instrument['id']); break
                value=matches[0]['value_um']
                if not isinstance(value,(int,float)) or not math.isfinite(value) or (f['characteristic']!='size' and value<0): raise ValueError('Evaluated results must be finite; geometric errors must be nonnegative.')
            state=classify(f,value,o['U_um'])
            event(f['id'],'MEASURE',instrument_id=instrument['id'],value_um=value,U_um=o['U_um'],coverage_factor=2,certificate=instrument['certificate'],reference=instrument['reference'],calibration_due=instrument['calibration_due'],status=state)
            final=dict(feature_id=f['id'],characteristic=f['characteristic'],instrument_id=instrument['id'],technique=instrument['technique'],value_um=value,U_um=o['U_um'],status=state,reason='Uncertainty interval decision',stage_xyz_mm=stage_xyz)
            if state!='Indeterminate': break
            previous=o
        event(f['id'],'COMPLETE' if final['status'] in ['Conforming','Nonconforming'] else 'REVIEW_REQUIRED',status=final['status'])
        results.append(final)
    return dict(version=VERSION,mode='SIMULATION' if observations is None else 'IMPORTED SCALARS / SIMULATED REGISTRATION',features=features,instruments=instruments,policy=dict(seed=seed,scenario=mode,ratio=ratio,registration_u_um=registration_u,environment_u_um=environment_u,localization_error_mm=localization_error_mm,evaluation_date=day.isoformat()),results=results,candidates=candidates,events=events)
