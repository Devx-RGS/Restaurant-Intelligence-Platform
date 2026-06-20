-- Group restaurants into price buckets and compare ratings
WITH pricing_buckets AS (
    SELECT 
        name,
        rate,
        votes,
        cost_for_two,
        CASE 
            WHEN cost_for_two <= 300 THEN '1. Budget (<=300)'
            WHEN cost_for_two BETWEEN 301 AND 600 THEN '2. Mid-Range (301-600)'
            WHEN cost_for_two BETWEEN 601 AND 1000 THEN '3. Premium (601-1000)'
            ELSE '4. Luxury (>1000)'
        END AS price_category
    FROM restaurants
)

SELECT 
    price_category,
    COUNT(*) AS total_restaurants,
    ROUND(AVG(rate)::numeric, 2) AS average_rating,
    ROUND(AVG(votes)::numeric, 0) AS average_votes
FROM pricing_buckets
GROUP BY price_category
ORDER BY price_category;
