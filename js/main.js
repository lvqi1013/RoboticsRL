/**
 * TurtleBot4 Navigation Experiment Results
 * 表格数据交互与美化逻辑
 */

document.addEventListener('DOMContentLoaded', () => {
    initMethodTabs();
    initTableHighlights();
    initSmoothScroll();
    initMetricTooltips();
});

/**
 * 方法标签切换
 */
function initMethodTabs() {
    const tabGroups = document.querySelectorAll('.method-tabs');
    tabGroups.forEach(group => {
        const tabs = group.querySelectorAll('.method-tab');
        const targetId = group.dataset.target;
        const targetCards = document.querySelectorAll(`.card[data-method]`);

        tabs.forEach(tab => {
            tab.addEventListener('click', () => {
                // 更新标签激活状态
                tabs.forEach(t => t.classList.remove('active'));
                tab.classList.add('active');

                const method = tab.dataset.method;

                // 过滤目标卡片
                if (targetId) {
                    const targetSection = document.getElementById(targetId);
                    if (targetSection) {
                        const cards = targetSection.querySelectorAll('.card[data-method]');
                        cards.forEach(card => {
                            if (method === 'all' || card.dataset.method === method) {
                                card.style.display = '';
                                card.style.animation = 'fadeInUp 0.3s ease forwards';
                            } else {
                                card.style.display = 'none';
                            }
                        });
                    }
                }

                // 过滤表格行
                targetCards.forEach(card => {
                    if (method === 'all' || card.dataset.method === method) {
                        card.style.display = '';
                    } else {
                        card.style.display = 'none';
                    }
                });
            });
        });
    });
}

/**
 * 表格数值高亮：自动标记每列的最佳值和最差值
 */
function initTableHighlights() {
    const tables = document.querySelectorAll('.data-table.auto-highlight');
    tables.forEach(table => {
        const rows = table.querySelectorAll('tbody tr');
        if (rows.length < 2) return;

        const colCount = rows[0].querySelectorAll('td').length;

        // 从第2列开始（第1列通常是标签列）
        for (let col = 1; col < colCount; col++) {
            let values = [];
            rows.forEach((row, rowIdx) => {
                const cell = row.querySelectorAll('td')[col];
                if (!cell) return;
                const val = parseFloat(cell.textContent);
                if (!isNaN(val)) {
                    values.push({ val, rowIdx, cell });
                }
            });

            if (values.length < 2) continue;

            // 判断该列是否为"越大越好"的指标
            const headerCell = table.querySelector(`thead th:nth-child(${col + 1})`);
            const headerText = headerCell ? headerCell.textContent.toLowerCase() : '';
            const higherBetter = ['r2', 'in_bounds_rate', 'valid_subgoal_rate',
                'reachable_segment_rate', 'progress_rate', 'mean_progress',
                'success_rate', 'accuracy', 'precision', 'recall', 'f1'
            ].some(k => headerText.includes(k));

            // 排序
            values.sort((a, b) => a.val - b.val);
            const best = higherBetter ? values[values.length - 1] : values[0];
            const worst = higherBetter ? values[0] : values[values.length - 1];

            // 高亮最佳值
            if (best && best.cell) {
                // 如果有多个相同最佳值，都高亮
                values.forEach(v => {
                    if (v.val === best.val && v.cell) {
                        v.cell.classList.add('best-value');
                    }
                });
            }
        }
    });
}

/**
 * 平滑滚动到锚点
 */
function initSmoothScroll() {
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
        });
    });
}

/**
 * 指标缩写提示
 */
function initMetricTooltips() {
    const tooltipData = {
        'MSE': 'Mean Squared Error — 均方误差，越小越好',
        'RMSE': 'Root Mean Squared Error — 均方根误差，越小越好',
        'MAE': 'Mean Absolute Error — 平均绝对误差，越小越好',
        'R²': 'R-squared — 决定系数，越接近1越好',
        'MDE': 'Maximum Distance Error — 最大距离误差，越小越好',
        'Median DE': 'Median Distance Error — 中位数距离误差，越小越好',
        'In-Bounds Rate': '子目标在可通行区域内的比例，越接近1越好',
        'Valid Subgoal Rate': '有效子目标比例，越接近1越好',
        'Obstacle Hit Rate': '障碍物碰撞率，越小越好',
        'Reachable Segment Rate': '可到达段比例，越接近1越好',
        'Progress Rate': '进度比例，越接近1越好',
        'Mean Progress': '平均进度值，越大越好'
    };

    document.querySelectorAll('.metric-tag').forEach(tag => {
        const abbr = tag.querySelector('.metric-abbr');
        if (!abbr) return;
        const key = abbr.textContent.trim();
        if (tooltipData[key]) {
            tag.setAttribute('title', tooltipData[key]);
        }
    });
}
