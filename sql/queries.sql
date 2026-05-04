SELECT
    unit_id,
    last_cycle,
    true_rul,
    predicted_rul,
    absolute_error,
    risk_level
FROM predictions
WHERE risk_level = 'HIGH'
ORDER BY predicted_rul ASC;


SELECT
    risk_level,
    COUNT(*) AS nombre_moteurs,
    ROUND(AVG(predicted_rul), 1) AS rul_moyen,
    ROUND(AVG(absolute_error), 1) AS erreur_moyenne
FROM predictions
GROUP BY risk_level
ORDER BY
    CASE risk_level
        WHEN 'HIGH' THEN 1
        WHEN 'MEDIUM' THEN 2
        WHEN 'LOW' THEN 3
    END;


SELECT
    unit_id,
    true_rul,
    predicted_rul,
    absolute_error,
    risk_level,
    CASE
        WHEN predicted_rul > true_rul THEN 'Surestimation'
        ELSE 'Sous-estimation'
    END AS type_erreur
FROM predictions
ORDER BY absolute_error DESC
LIMIT 10;


SELECT
    risk_level,
    ROUND(AVG(true_rul), 1) AS rul_reel_moyen,
    ROUND(AVG(predicted_rul), 1) AS rul_predit_moyen,
    ROUND(AVG(predicted_rul) - AVG(true_rul), 1) AS biais_moyen
FROM predictions
GROUP BY risk_level
ORDER BY
    CASE risk_level
        WHEN 'HIGH' THEN 1
        WHEN 'MEDIUM' THEN 2
        WHEN 'LOW' THEN 3
    END;


SELECT
    model,
    evaluation,
    ROUND(rmse, 2) AS rmse,
    ROUND(mae, 2) AS mae,
    ROUND(nasa_score, 2) AS nasa_score
FROM metrics
WHERE evaluation = 'validation_all_cycles'
ORDER BY rmse ASC;


SELECT
    unit_id,
    cycle,
    sensor_2,
    sensor_2_rollmean_5,
    sensor_2_delta
FROM sensor_readings
WHERE unit_id = 1
ORDER BY cycle ASC;


SELECT
    unit_id,
    true_rul,
    predicted_rul,
    ROUND(predicted_rul - true_rul, 1) AS surestimation
FROM predictions
WHERE predicted_rul > true_rul
ORDER BY surestimation DESC
LIMIT 10;


SELECT
    COUNT(*) AS total_moteurs,
    SUM(CASE WHEN risk_level = 'HIGH' THEN 1 ELSE 0 END) AS moteurs_critiques,
    SUM(CASE WHEN risk_level = 'MEDIUM' THEN 1 ELSE 0 END) AS moteurs_a_surveiller,
    SUM(CASE WHEN risk_level = 'LOW' THEN 1 ELSE 0 END) AS moteurs_stables,
    ROUND(AVG(predicted_rul), 1) AS rul_moyen_predit,
    ROUND(AVG(absolute_error), 1) AS erreur_moyenne
FROM predictions;


SELECT
    p.risk_level,
    ROUND(AVG(s.sensor_2), 3) AS avg_sensor_2,
    ROUND(AVG(s.sensor_3), 3) AS avg_sensor_3,
    ROUND(AVG(s.sensor_4), 3) AS avg_sensor_4,
    ROUND(AVG(s.sensor_11), 3) AS avg_sensor_11,
    ROUND(AVG(s.sensor_15), 3) AS avg_sensor_15
FROM predictions p
JOIN sensor_readings s
    ON p.unit_id = s.unit_id
WHERE p.risk_level = 'HIGH'
GROUP BY p.risk_level;