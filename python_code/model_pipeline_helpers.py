"""
SMARTRENT — ชุดฟังก์ชันเตรียมข้อมูล สร้างโมเดล และวัดผล (Phase 2 Baseline)
รวบรวมขั้นตอน Data Preprocessing, Training, Evaluation และการตรวจสอบโมเดลที่บันทึกไว้
"""

from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import joblib
from sklearn.base import clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits

SEED = 2026

# 18 ตัวแปรทำนาย (Features) ก่อนทำ One-Hot Encoding และเติมค่าว่าง
ROOM = [
    'log_size', 'floor_no', 'bedrooms', 'is_duplex', 'n_reported_room_amenities',
    'air_conditioner', 'digital_door_lock', 'furnished', 'hot_tub', 'in_room_wifi',
    'kitchen_hood', 'kitchen_stove', 'phone', 'refrigerator', 'tv', 'washer', 'water_heater'
]
CAT = ['unit_type']
FEATURES = ROOM + CAT


def preprocessor(num, cat):
    """สร้าง Pipeline เตรียมข้อมูล: เติมค่าว่าง + ปรับสเกลตัวเลข + One-Hot Encoding ประเภทห้อง"""
    return ColumnTransformer([
        ('numeric', Pipeline([
            ('impute', SimpleImputer(strategy='median', add_indicator=True)),
            ('scale', StandardScaler()),
        ]), num),
        ('categorical', Pipeline([
            ('impute', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
        ]), cat),
    ], sparse_threshold=0)


def model_pipeline(model, num, cat, log_target=True):
    """รวมขั้นเตรียมข้อมูลและตัวโมเดลเข้าด้วยกัน พร้อมแปลงค่าเช่าเป็น log1p ระหว่างฝึก"""
    reg = (
        TransformedTargetRegressor(regressor=model, func=np.log1p, inverse_func=np.expm1)
        if log_target
        else model
    )
    return Pipeline([('preprocess', preprocessor(num, cat)), ('regressor', reg)])


def metrics(actual, predicted, projects):
    """คำนวณตัวชี้วัดความคลาดเคลื่อน: MAE, RMSE, R2, MAPE, Within 20%, และ Project Macro MAE"""
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    if not np.isfinite(predicted).all():
        raise ValueError('Non-finite model prediction')
    e = np.abs(actual - predicted)
    group_mae = pd.DataFrame({'project': np.asarray(projects), 'ae': e}).groupby('project').ae.mean()
    return {
        'MAE_THB': float(e.mean()),
        'RMSE_THB': float(mean_squared_error(actual, predicted) ** 0.5),
        'R2': float(r2_score(actual, predicted)),
        'MAPE_pct': float((e / actual).mean() * 100),
        'within20_pct': float((e / actual <= 0.2).mean() * 100),
        'project_macro_MAE_THB': float(group_mae.mean()),
    }


def load_data(root):
    """โหลดไฟล์ train / validation / test และรวม train + validation สำหรับเทรนรอบสุดท้าย"""
    root = Path(root)
    sets = {
        name: pd.read_csv(root / 'data' / f'{name}.csv', float_precision='round_trip')
        for name in ['train', 'validation', 'test']
    }
    train, val, test = sets['train'], sets['validation'], sets['test']
    dev = pd.concat([train, val], ignore_index=True)
    assert [len(x) for x in [train, val, test, dev]] == [4217, 1408, 800, 5625]
    for d in [train, val, test]:
        assert d.room_id.is_unique
        assert d.rental.gt(0).all()
        assert set(FEATURES).issubset(d.columns)
    ids = [set(x.room_id) for x in [train, val, test]]
    groups = [set(x.project_id) for x in [train, val, test]]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
    return train, val, test, dev


def verify_saved(root):
    """โหลดโมเดลที่บันทึกไว้มาทำนายชุด Test 800 รายการ เพื่อเช็กว่าตรงกับผลที่รายงานจริง"""
    root = Path(root)
    train, val, test, dev = load_data(root)
    model = joblib.load(root / 'models' / 'phase2_ridge_rent_model.joblib')
    assert list(model.named_steps['preprocess'].feature_names_in_) == FEATURES
    p = model.predict(test[FEATURES])
    ref = pd.read_csv(root / 'results' / 'test_predictions.csv').set_index('room_id').loc[test.room_id]
    np.testing.assert_allclose(p, ref.predicted_asking_rent_THB, rtol=1e-9, atol=1e-6)
    np.testing.assert_allclose(test.rental, ref.actual_asking_rent_THB)
    return pd.DataFrame([{'model': 'Saved Ridge (Phase 2)', **metrics(test.rental, p, test.project_id)}])


def reproduce_ridge(root, output):
    """ฝึกโมเดล Ridge Regression (alpha=10) ซ้ำตั้งแต่ต้น แล้วบันทึกผลไว้ใน output"""
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    train, val, test, dev = load_data(root)
    with threadpool_limits(limits=2):
        candidate = model_pipeline(Ridge(alpha=10), ROOM, CAT).fit(train[FEATURES], train.rental)
        validation = metrics(val.rental, candidate.predict(val[FEATURES]), val.project_id)
        final = model_pipeline(Ridge(alpha=10), ROOM, CAT).fit(dev[FEATURES], dev.rental)
        p = final.predict(test[FEATURES])
    ref = pd.read_csv(root / 'results' / 'test_predictions.csv').set_index('room_id').loc[test.room_id]
    np.testing.assert_allclose(p, ref.predicted_asking_rent_THB, rtol=1e-8, atol=1e-5)
    result = pd.DataFrame([
        {'stage': 'Validation', 'model': 'Ridge', **validation},
        {'stage': 'Test', 'model': 'Ridge', **metrics(test.rental, p, test.project_id)},
    ])
    result.to_csv(output / 'ridge_results.csv', index=False)
    pd.DataFrame({
        'room_id': test.room_id,
        'actual_asking_rent_THB': test.rental,
        'predicted_asking_rent_THB': p,
    }).to_csv(output / 'ridge_predictions.csv', index=False)
    joblib.dump(final, output / 'ridge_retrained.joblib')
    return result


def reproduce_all(root, output):
    """ฝึกเปรียบเทียบทุกโมเดล (Median, Ridge, Random Forest, Gradient Boosting) ซ้ำทั้งหมด"""
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    train, val, test, dev = load_data(root)
    args = SimpleNamespace(output=output)
    with threadpool_limits(limits=2):
        candidates = {
            'Median baseline': model_pipeline(DummyRegressor(strategy='median'), ROOM, ['unit_type'], False),
            'Ridge log rent': model_pipeline(Ridge(alpha=10), ROOM, CAT),
            'Random Forest rooms': model_pipeline(
                RandomForestRegressor(n_estimators=160, min_samples_leaf=5, max_features=0.8, n_jobs=2, random_state=SEED),
                ROOM,
                ['unit_type'],
            ),
        }
        hgb = model_pipeline(
            HistGradientBoostingRegressor(max_iter=220, early_stopping=False, random_state=SEED),
            ROOM,
            CAT,
        )
        grid = {
            'regressor__regressor__learning_rate': [0.05, 0.1],
            'regressor__regressor__max_leaf_nodes': [15, 31],
            'regressor__regressor__l2_regularization': [1.0, 10.0],
        }
        search = GridSearchCV(
            hgb, grid, scoring='neg_mean_absolute_error', cv=GroupKFold(n_splits=3), n_jobs=1, error_score='raise'
        )
        search.fit(train[FEATURES], train.rental, groups=train.project_id)
        candidates['Gradient Boosting tuned'] = search.best_estimator_
        pd.DataFrame(search.cv_results_).to_csv(args.output / 'training_cv_search.csv', index=False)

        rows = []
        fitted = {}
        for name, est in candidates.items():
            fit = clone(est).fit(train[FEATURES], train.rental)
            prediction = fit.predict(val[FEATURES])
            rows.append({'model': name, **metrics(val.rental, prediction, val.project_id)})
            fitted[name] = fit
            print(name, round(rows[-1]['MAE_THB'], 2), flush=True)

        comparison = pd.DataFrame(rows).sort_values('MAE_THB').reset_index(drop=True)
        comparison.to_csv(args.output / 'validation_results.csv', index=False)
        selected = str(comparison.iloc[0]['model'])
        print(comparison.to_string(index=False))
        print('Selected by Validation MAE:', selected)

    final = clone(candidates[selected]).fit(dev[FEATURES], dev.rental)
    baseline = clone(candidates['Median baseline']).fit(dev[FEATURES], dev.rental)
    pred = final.predict(test[FEATURES])
    base_pred = baseline.predict(test[FEATURES])
    final_rows = [
        {'model': selected, **metrics(test.rental, pred, test.project_id)},
        {'model': 'Median baseline', **metrics(test.rental, base_pred, test.project_id)},
    ]
    pd.DataFrame(final_rows).to_csv(args.output / 'test_results.csv', index=False)

    prediction_table = test[['room_id', 'project_id', 'unit_type', 'size_sqm', 'floor_no', 'rental']].copy()
    prediction_table = prediction_table.rename(columns={'rental': 'actual_asking_rent_THB'})
    prediction_table['predicted_asking_rent_THB'] = pred
    prediction_table['absolute_error_THB'] = np.abs(test.rental.to_numpy() - pred)
    prediction_table['absolute_error_pct'] = (
        prediction_table.absolute_error_THB / prediction_table.actual_asking_rent_THB * 100
    )
    prediction_table.to_csv(args.output / 'test_predictions.csv', index=False, encoding='utf-8-sig')
    prediction_table.sample(min(8, len(prediction_table)), random_state=SEED).to_csv(
        args.output / 'demo_examples.csv', index=False, encoding='utf-8-sig'
    )
    per_project = (
        prediction_table.groupby(['project_id'])
        .agg(listings=('room_id', 'size'), MAE_THB=('absolute_error_THB', 'mean'))
        .reset_index()
        .sort_values('MAE_THB', ascending=False)
    )
    per_project.to_csv(args.output / 'test_error_by_project.csv', index=False, encoding='utf-8-sig')
    joblib.dump(final, args.output / 'rent_model.joblib')

    sample = test[FEATURES].head(1)
    sample.to_csv(args.output / 'example_model_input.csv', index=False)
    loaded = joblib.load(args.output / 'rent_model.joblib')
    np.testing.assert_allclose(loaded.predict(sample), final.predict(sample))

    print('Test results (THB/month):')
    print(pd.DataFrame(final_rows).to_string(index=False))
    print(
        'Within 20% count:',
        int((np.abs(test.rental.to_numpy() - pred) / test.rental.to_numpy() <= 0.2).sum()),
        '/',
        len(test),
    )
    return pd.DataFrame(final_rows)
