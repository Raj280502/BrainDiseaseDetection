import React from 'react';

const DashboardPage = () => {
  return (
    // Main container with top padding to account for the fixed navbar
    <div className="pt-20 min-h-screen bg-gray-100 flex">
      {/* --- Sidebar --- */}
      <aside className="w-64 bg-white shadow-md">
        <div className="p-6">
          <h2 className="text-xl font-bold text-gray-800 mb-6">Menu</h2>
          <nav>
            <ul>
              <li>
                <a 
                  href="#" 
                  className="block py-2 px-4 rounded bg-blue-100 text-blue-700 font-semibold"
                >
                  Detections
                </a>
              </li>
              <li className="mt-2">
                <a 
                  href="#" 
                  className="block py-2 px-4 rounded hover:bg-gray-200 text-gray-600"
                >
                  Report
                </a>
              </li>
            </ul>
          </nav>
        </div>
      </aside>

      {/* --- Main Content --- */}
      <main className="flex-1 p-10">
        <h1 className="text-3xl font-bold text-gray-800">Dashboard</h1>
        <p className="mt-4 text-gray-600">Welcome to your dashboard. Your content will be displayed here.</p>
        
        {/* We will enhance this section later */}
        <div className="mt-8 p-6 bg-white rounded-lg shadow-md">
            <h2 className="text-xl font-semibold">Overview</h2>
            {/* Placeholder for future content */}
        </div>
      </main>
    </div>
  );
};

export default DashboardPage;