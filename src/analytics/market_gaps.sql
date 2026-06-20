-- Split comma-separated cuisines into separate rows
WITH split_cuisines AS (
    SELECT 
        name,
        location,
        votes,
        TRIM(unnest(string_to_array(cuisines, ','))) AS single_cuisine
    FROM restaurants
),

-- Calculate restaurant count and total votes per cuisine per location
locality_stats AS (
    SELECT 
        location,
        single_cuisine,
        COUNT(*) AS total_restaurants,
        SUM(votes) AS total_demand_votes
    FROM split_cuisines
    GROUP BY location, single_cuisine
)

-- Find high-demand, low-supply combinations (Market Gaps)
SELECT 
    location,
    single_cuisine,
    total_restaurants AS supply,
    total_demand_votes AS demand,
    ROUND(total_demand_votes / total_restaurants) AS votes_per_restaurant
FROM locality_stats
WHERE total_restaurants BETWEEN 1 AND 5
  AND total_demand_votes > 1000
ORDER BY votes_per_restaurant DESC
LIMIT 10;
