import React, { useEffect, useState } from 'react';

function CaseList() {
  const [cases, setCases] = useState([]);

  useEffect(() => {
    fetch('http://localhost:5000/api/cases')
      .then(res => res.json())
      .then(data => setCases(data))
      .catch(err => console.error('Error fetching cases:', err));
  }, []);

  return (
    <div>
      <h2>Case List</h2>
      <ul>
        {cases.map((c, idx) => (
          <li key={idx}>
            {c.name} (Status: {c.status}, Created: {c.created_at})
          </li>
        ))}
      </ul>
    </div>
  );
}

export default CaseList;