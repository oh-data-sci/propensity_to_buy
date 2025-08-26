SELECT
    COUNT(*)                                              AS num_records,
    COUNT(DISTINCT lead_id)                               AS num_customers,
    COUNT(DISTINCT postcode)                              AS num_locations,
    COUNT(DISTINCT REGEXP_EXTRACT(postcode,'(.*?)\d',1))  AS num_postcode_areas,
    COUNT(DISTINCT EXTRACT(YEAR FROM date_of_birth))      AS num_years_of_birth,
    MIN(date_of_birth)                                    AS oldest_birthday,
    MAX(date_of_birth)                                    AS youngest_birthday,
    COUNT(DISTINCT company_industry)                      AS num_industries,
    COUNT(DISTINCT step_reached_in_website)               AS num_steps,
    COUNT(DISTINCT how_did_you_hear_about_us)             AS num_sources, 
    MIN(salary)                                           AS min_salary,
    MAX(salary)                                           AS max_salary,
    AVG(salary)                                           AS avg_salary,
    MEDIAN(salary)                                        AS med_salary,
    MEDIAN(n_engaged_minutes)                             AS med_engaged_length
FROM sales
;

SELECT
    has_placed_order                                      AS has_placed_order,
    COUNT(*)                                              AS num_records,
    COUNT(DISTINCT lead_id)                               AS num_customers,
    COUNT(DISTINCT postcode)                              AS num_locations,
    COUNT(DISTINCT REGEXP_EXTRACT(postcode,'(.*?)\d',1))  AS num_postcode_areas,
    COUNT(DISTINCT EXTRACT(YEAR FROM date_of_birth))      AS num_years_of_birth,
    MIN(date_of_birth)                                    AS oldest_birthday,
    MAX(date_of_birth)                                    AS youngest_birthday,
    COUNT(DISTINCT company_industry)                      AS num_industries,
    COUNT(DISTINCT step_reached_in_website)               AS num_steps,
    COUNT(DISTINCT how_did_you_hear_about_us)             AS num_sources, 
    MIN(salary)                                           AS min_salary,
    MAX(salary)                                           AS max_salary,
    AVG(salary)                                           AS avg_salary,
    MEDIAN(salary)                                        AS med_salary,
    MEDIAN(n_engaged_minutes)                             AS med_engaged_length
FROM sales
GROUP BY has_placed_order
;