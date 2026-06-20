-- Rank restaurants within each location by rating and votes
WITH ranked_restaurants AS (
    SELECT 
        name,
        location,
        rate,
        votes,
        cost_for_two,
        rest_type,
        cuisines,
        ROW_NUMBER() OVER (
            PARTITION BY location 
            ORDER BY rate DESC, votes DESC
        ) AS rank_in_area
    FROM restaurants
)

-- Show only the top-ranked restaurant from each location
SELECT 
    location,
    name,
    rate,
    votes,
    cost_for_two,
    rest_type,
    cuisines
FROM ranked_restaurants
WHERE rank_in_area = 1
ORDER BY rate DESC, votes DESC
LIMIT 15;
