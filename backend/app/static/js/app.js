// DOM Elements
const statusEl = document.getElementById("status");
const tableBody = document.getElementById("table-body");
const generateBtn = document.getElementById("generate-data-btn");
const etlBtn = document.getElementById("run-etl-btn");

const statGroups = document.getElementById("stat-groups");
const statTxns = document.getElementById("stat-txns");
const statAmount = document.getElementById("stat-amount");

/**
 * Fetch và hiển thị stats từ API
 */
async function fetchStats() {
  try {
    const res = await fetch("/stats");
    const data = await res.json();
    tableBody.innerHTML = "";
    
    let totalGroups = 0;
    let totalTxns = 0;
    let totalAmt = 0;
    
    for (const item of data.items || []) {
      totalGroups += 1;
      totalTxns += item.txn_count;
      totalAmt += item.total_amount;
      
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td><span class="pill">${item.country}</span></td>
        <td>${item.category}</td>
        <td>${item.txn_count.toLocaleString()}</td>
        <td>${item.total_amount.toLocaleString(undefined, { 
          minimumFractionDigits: 2, 
          maximumFractionDigits: 2 
        })}</td>
      `;
      tableBody.appendChild(tr);
    }
    
    statGroups.textContent = totalGroups.toLocaleString();
    statTxns.textContent = totalTxns.toLocaleString();
    statAmount.textContent = totalAmt.toLocaleString(undefined, { 
      minimumFractionDigits: 2, 
      maximumFractionDigits: 2 
    });
    
    statusEl.textContent = "Stats loaded.";
  } catch (e) {
    console.error("Error fetching stats:", e);
    statusEl.textContent = "Failed to load stats.";
  }
}

/**
 * Xử lý sự kiện click nút Generate Data
 */
generateBtn.addEventListener("click", async () => {
  statusEl.textContent = "Starting data generation (uploading to S3)...";
  generateBtn.disabled = true;
  
  try {
    const res = await fetch("/data/generate", { method: "POST" });
    const data = await res.json();
    
    statusEl.textContent = `Data generation started: ${data.config.total_rows.toLocaleString()} rows, ${data.config.estimated_chunks} chunks. Check server logs for progress.`;
    
    // Check S3 files after 10 seconds
    setTimeout(async () => {
      try {
        const s3Res = await fetch("/s3/list");
        const s3Data = await s3Res.json();
        statusEl.textContent = `Data generation in progress. ${s3Data.total_files} files uploaded to S3.`;
      } catch (e) {
        console.error("Error checking S3 files:", e);
      }
    }, 10000);
  } catch (e) {
    console.error("Error starting data generation:", e);
    statusEl.textContent = "Failed to start data generation.";
  } finally {
    generateBtn.disabled = false;
  }
});

/**
 * Xử lý sự kiện click nút Run ETL
 */
etlBtn.addEventListener("click", async () => {
  statusEl.textContent = "Starting ETL job với tất cả các phương pháp xử lý dữ liệu (this may take a few minutes for 2M rows)...";
  etlBtn.disabled = true;
  
  try {
    const res = await fetch("/etl/run", { method: "POST" });
    const data = await res.json();
    
    statusEl.textContent = "ETL started với 7 phương pháp xử lý. Waiting 30s before first stats refresh...";
    setTimeout(fetchStats, 30000);
  } catch (e) {
    console.error("Error starting ETL:", e);
    statusEl.textContent = "Failed to start ETL.";
  } finally {
    etlBtn.disabled = false;
  }
});

// Chart instances
let charts = {};

/**
 * Fetch và hiển thị component evaluation
 */
async function fetchComponentEvaluation() {
  try {
    const res = await fetch("/api/components");
    const data = await res.json();
    const componentsGrid = document.getElementById("components-grid");
    
    if (!componentsGrid) return;
    
    componentsGrid.innerHTML = "";
    
    for (const component of data.components || []) {
      const card = document.createElement("div");
      card.className = "component-card";
      card.innerHTML = `
        <div class="component-header">
          <div class="component-name">${component.name}</div>
          <div class="component-status ${component.status}">${component.status}</div>
        </div>
        <div class="component-score">${component.score}/100</div>
        <div class="component-description">${component.description}</div>
        <div class="component-details">${component.details}</div>
      `;
      componentsGrid.appendChild(card);
    }
  } catch (e) {
    console.error("Error fetching component evaluation:", e);
  }
}

/**
 * Fetch và render charts
 */
async function fetchAndRenderCharts() {
  try {
    // Country charts
    const countryRes = await fetch("/api/charts/by-country");
    const countryData = await countryRes.json();
    
    // Transactions by Country
    const ctxCountryTxns = document.getElementById("chart-country-txns");
    if (ctxCountryTxns && charts.countryTxns) {
      charts.countryTxns.destroy();
    }
    if (ctxCountryTxns) {
      charts.countryTxns = new Chart(ctxCountryTxns, {
        type: "bar",
        data: {
          labels: countryData.labels || [],
          datasets: [{
            label: "Transactions",
            data: countryData.txn_data || [],
            backgroundColor: "rgba(59, 130, 246, 0.6)",
            borderColor: "rgba(59, 130, 246, 1)",
            borderWidth: 1
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: true,
          plugins: {
            legend: {
              labels: { color: "#e5e7eb" }
            }
          },
          scales: {
            y: {
              beginAtZero: true,
              ticks: { color: "#9ca3af" },
              grid: { color: "#1e293b" }
            },
            x: {
              ticks: { color: "#9ca3af" },
              grid: { color: "#1e293b" }
            }
          }
        }
      });
    }
    
    // Amount by Country
    const ctxCountryAmount = document.getElementById("chart-country-amount");
    if (ctxCountryAmount && charts.countryAmount) {
      charts.countryAmount.destroy();
    }
    if (ctxCountryAmount) {
      charts.countryAmount = new Chart(ctxCountryAmount, {
        type: "bar",
        data: {
          labels: countryData.labels || [],
          datasets: [{
            label: "Total Amount",
            data: countryData.amount_data || [],
            backgroundColor: "rgba(34, 197, 94, 0.6)",
            borderColor: "rgba(34, 197, 94, 1)",
            borderWidth: 1
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: true,
          plugins: {
            legend: {
              labels: { color: "#e5e7eb" }
            }
          },
          scales: {
            y: {
              beginAtZero: true,
              ticks: { color: "#9ca3af" },
              grid: { color: "#1e293b" }
            },
            x: {
              ticks: { color: "#9ca3af" },
              grid: { color: "#1e293b" }
            }
          }
        }
      });
    }
    
    // Category charts
    const categoryRes = await fetch("/api/charts/by-category");
    const categoryData = await categoryRes.json();
    
    // Transactions by Category
    const ctxCategoryTxns = document.getElementById("chart-category-txns");
    if (ctxCategoryTxns && charts.categoryTxns) {
      charts.categoryTxns.destroy();
    }
    if (ctxCategoryTxns) {
      charts.categoryTxns = new Chart(ctxCategoryTxns, {
        type: "doughnut",
        data: {
          labels: categoryData.labels || [],
          datasets: [{
            label: "Transactions",
            data: categoryData.txn_data || [],
            backgroundColor: [
              "rgba(59, 130, 246, 0.6)",
              "rgba(34, 197, 94, 0.6)",
              "rgba(251, 191, 36, 0.6)",
              "rgba(239, 68, 68, 0.6)",
              "rgba(168, 85, 247, 0.6)",
              "rgba(236, 72, 153, 0.6)"
            ],
            borderWidth: 1
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: true,
          plugins: {
            legend: {
              position: "bottom",
              labels: { color: "#e5e7eb" }
            }
          }
        }
      });
    }
    
    // Amount by Category
    const ctxCategoryAmount = document.getElementById("chart-category-amount");
    if (ctxCategoryAmount && charts.categoryAmount) {
      charts.categoryAmount.destroy();
    }
    if (ctxCategoryAmount) {
      charts.categoryAmount = new Chart(ctxCategoryAmount, {
        type: "pie",
        data: {
          labels: categoryData.labels || [],
          datasets: [{
            label: "Total Amount",
            data: categoryData.amount_data || [],
            backgroundColor: [
              "rgba(59, 130, 246, 0.6)",
              "rgba(34, 197, 94, 0.6)",
              "rgba(251, 191, 36, 0.6)",
              "rgba(239, 68, 68, 0.6)",
              "rgba(168, 85, 247, 0.6)",
              "rgba(236, 72, 153, 0.6)"
            ],
            borderWidth: 1
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: true,
          plugins: {
            legend: {
              position: "bottom",
              labels: { color: "#e5e7eb" }
            }
          }
        }
      });
    }
  } catch (e) {
    console.error("Error fetching charts:", e);
  }
}

/**
 * Fetch và hiển thị evaluation data
 */
async function fetchEvaluation() {
  try {
    const res = await fetch("/api/evaluation");
    const data = await res.json();
    
    // Overall evaluation
    const overallBody = document.getElementById("evaluation-overall-body");
    if (overallBody && data.overall) {
      const overall = data.overall;
      overallBody.innerHTML = `
        <tr>
          <td class="metric-label">Total Groups</td>
          <td class="metric-value highlight">${overall.total_groups.toLocaleString()}</td>
        </tr>
        <tr>
          <td class="metric-label">Total Transactions</td>
          <td class="metric-value highlight">${overall.total_txns.toLocaleString()}</td>
        </tr>
        <tr>
          <td class="metric-label">Total Amount</td>
          <td class="metric-value highlight">${overall.total_amount.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</td>
        </tr>
        <tr>
          <td class="metric-label">Average Amount</td>
          <td class="metric-value">${overall.avg_amount.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</td>
        </tr>
        <tr>
          <td class="metric-label">Max Amount</td>
          <td class="metric-value">${overall.max_amount.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</td>
        </tr>
        <tr>
          <td class="metric-label">Min Amount</td>
          <td class="metric-value">${overall.min_amount.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</td>
        </tr>
        <tr>
          <td class="metric-label">Countries</td>
          <td class="metric-value">${overall.country_count}</td>
        </tr>
        <tr>
          <td class="metric-label">Categories</td>
          <td class="metric-value">${overall.category_count}</td>
        </tr>
      `;
    }
    
    // Country evaluation
    const countryBody = document.getElementById("evaluation-country-body");
    if (countryBody && data.by_country) {
      countryBody.innerHTML = "";
      for (const item of data.by_country) {
        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td><span class="pill">${item.country}</span></td>
          <td>${item.group_count.toLocaleString()}</td>
          <td>${item.total_txns.toLocaleString()}</td>
          <td class="metric-value highlight">${item.total_amount.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</td>
          <td>${item.avg_amount.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</td>
          <td>${item.max_amount.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</td>
          <td>${item.min_amount.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</td>
        `;
        countryBody.appendChild(tr);
      }
    }
    
    // Category evaluation
    const categoryBody = document.getElementById("evaluation-category-body");
    if (categoryBody && data.by_category) {
      categoryBody.innerHTML = "";
      for (const item of data.by_category) {
        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td>${item.category}</td>
          <td>${item.group_count.toLocaleString()}</td>
          <td>${item.total_txns.toLocaleString()}</td>
          <td class="metric-value highlight">${item.total_amount.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</td>
          <td>${item.avg_amount.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</td>
          <td>${item.max_amount.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</td>
          <td>${item.min_amount.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</td>
        `;
        categoryBody.appendChild(tr);
      }
    }
  } catch (e) {
    console.error("Error fetching evaluation:", e);
  }
}

/**
 * Tab switching functionality
 */
function initTabs() {
  const tabButtons = document.querySelectorAll(".tab-btn");
  const tabContents = document.querySelectorAll(".tab-content");
  
  tabButtons.forEach(btn => {
    btn.addEventListener("click", () => {
      const targetTab = btn.getAttribute("data-tab");
      
      // Remove active class from all buttons and contents
      tabButtons.forEach(b => b.classList.remove("active"));
      tabContents.forEach(c => c.classList.remove("active"));
      
      // Add active class to clicked button and corresponding content
      btn.classList.add("active");
      const targetContent = document.getElementById(`tab-${targetTab}`);
      if (targetContent) {
        targetContent.classList.add("active");
      }
    });
  });
}

/**
 * Refresh all data
 */
async function refreshAll() {
  await Promise.all([
    fetchStats(),
    fetchComponentEvaluation(),
    fetchEvaluation(),
    fetchAndRenderCharts()
  ]);
}

// Initialize: Fetch all data on page load
refreshAll();
initTabs();

// Auto-refresh all data every 60 seconds
setInterval(refreshAll, 60000);

