**构建基于大语言模型的期货基本面分析及成本影响评估工具之综合指南**
==================================================================

**I. 基本面分析的战略性数据采集**
---------------------------------

构建一个强大分析工具的基础在于全面、可靠且经济高效的数据源。本章节聚焦于识别和获取对期货商品基本面分析至关重要的数据，旨在为大语言模型
(LLM) 驱动的分析工具建立一个综合数据管道。

### **A. 识别关键的免费及低成本新闻与数据源**

此分节详述各类数据源，强调其与商品期货分析的相关性以及获取途径。

#### **1. 政府及超国家机构报告**

这些报告是理解关键商品市场供需动态的基础，它们提供来自权威机构的结构化数据和预测。

-   **美国农业部全球农业供需预测 (USDA WASDE):**
    > 此报告每月发布，提供美国及全球小麦、大米、粗粮（玉米、大麦、高粱、燕麦）、油籽（大豆、油菜籽、棕榈油）和棉花的全面供需预测。报告还涵盖美国食糖、肉类、禽类、蛋类和牛奶的供需情况
    > [^1]。这些报告在严格保密的情况下编写，并在特定时间发布，其发布常常引发市场即时波动
    > [^2]。历史数据可追溯至1973年
    > [^1]。了解其发布时间表（例如^1^中提及的2025年发布日期）对于保证信息的及时性至关重要。

-   **美国能源信息署石油状况周报 (EIA WPSR):**
    > 由美国能源信息署发布，此报告详细说明美国原油储备、产量以及汽油、取暖油和柴油等精炼产品的库存水平
    > [^3]。它是石油需求的关键指标，能够影响价格趋势
    > [^3]。报告包含进口、炼油厂输入和产品需求等数据
    > [^5]。EIA通过API、Excel插件和批量文件提供开放数据访问 [^4]。

-   **其他政府来源 (例如，澳大利亚工业部关于铁矿石的数据):**
    > 诸如《资源与能源季报》之类的报告提供铁矿石等商品的生产统计、出口量和价格展望
    > [^9]。特定地区的数据，如西澳大利亚州的铁矿石概况，则提供关于生产和出口的细化信息
    > [^10]。这些对于特定的工业品期货至关重要。

这些官方报告为基本面分析提供了基准。它们的结构化特性（尽管有时是PDF格式，需要提取，见第二节A部分）提供了可靠的量化数据，可用于校准或验证从非结构化新闻中获得的洞察。然而，单一数据源往往存在局限性。例如，官方报告虽然可靠，但其发布具有周期性，可能无法捕捉到最新的市场动态。真正的分析力量来源于对多种数据类型的协同融合。将官方报告的结构化数据与来自金融API的实时非结构化新闻以及来自另类数据源的独特指标相结合，能够形成更全面、更及时的市场认知。例如，通过Sentinel
Hub获取的显示某关键大豆产区干旱的卫星图像 [^11]，可以结合最新的USDA
WASDE报告中的产量预测进行情境化分析，并通过金融API的实时新闻流监控其后续影响。大语言模型在综合处理这些多样化输入方面可以发挥关键作用。

#### **2. 专业商品数据提供商及交易所数据**

这包括主要商品生产商的直接生产报告和商品交易所的数据。

-   **主要矿业公司 (例如，淡水河谷, 必和必拓):** 诸如淡水河谷 [^13]
    > 和必和必拓 [^17]
    > 等公司会发布季度或定期生产和销售报告。这些报告提供铁矿石、铜和镍等商品产量的具体细节，通常包括已实现价格、成本数据（如淡水河谷的C1现金成本
    > [^15]）以及运营挑战（如降雨对淡水河谷产量的影响
    > [^13]）。这些报告通常在其投资者关系网站上以PDF格式提供。

-   **商品交易所 (例如，芝商所, 大连商品交易所 (DCE), 新加坡交易所
    > (SGX)):**

    -   **芝商所 (CME Group):**
        > 提供跨越农业、能源和金属等多种资产类别的广泛市场数据。他们提供历史数据平台、分析工具
        > (QuikStrike) 和实时数据流
        > [^21]。虽然某些访问可能是付费的，但他们也提供报告和市场洞察。

    -   **大连商品交易所 (DCE):**
        > 是铁矿石期货的重要交易所。提供市场统计数据，以及大豆和铁矿石等商品的每日价格
        > [^23]。部分英文数据或可获取 [^24]。第三方提供商如Barchart [^25]
        > 或Commodities-API [^26]
        > 可能通过API提供DCE数据，但成本各异。英国《金融时报》[^27]
        > 和财智莫尼 [^28] 也报道DCE铁矿石价格。

    -   **新加坡交易所 (SGX):**
        > 对铁矿石衍生品市场非常重要。提供各种铁矿石产品（例如，62%品位铁矿石粉矿，58%品位铁矿石粉矿）的价格、图表和合约细则
        > [^29]。历史数据可在Investing.com等平台上找到 [^30]。

生产商报告能直接揭示关键参与者的供应面情况。交易所数据则提供价格发现、交易量和持仓量信息，这对于理解市场情绪和流动性至关重要。

#### **3. 金融新闻API**

这些API对于获取LLM将处理的及时、非结构化的新闻数据至关重要。许多API提供免费或低成本的层级，适合开发和初步部署。

-   **Marketstack:**
    > 提供实时和历史股票市场数据，覆盖全球。适合构建股票跟踪工具或需要市场洞察的开发者
    > [^31]。

-   **Alpha Vantage:**
    > 知名的免费金融数据API，提供实时和历史市场数据，包括期权。适用于多样化的金融指标获取
    > [^31]。

-   **Finnhub:** 提供免费、高质量的金融数据，包含实时股市动态 [^31]。

-   **IEX Cloud:**
    > 提供适用于将分析功能集成到Web应用程序以及供数据科学家使用的金融数据
    > [^31]。

-   **Twelve Data:** 广泛的金融市场覆盖，适合需要多资产数据访问的用户
    > [^31]。

-   **Polygon.io:**
    > 提供机构级别的股票、期权和加密货币数据访问，包括带有情感分析等功能的金融新闻API
    > [^32]。他们还提供Benzinga的分析师评级和财报API。

-   **NewsAPI.org:**
    > 为开发者提供免费层级（每日100次请求，文章延迟24小时），并提供付费计划以实时访问来自众多来源的文章
    > [^33]。

-   **NewsAPI.ai:**
    > 专注于提供带有元数据（如实体、主题和情感）的全球新闻内容。提供Python和Node.JS的SDK
    > [^34]。

-   **Apify (用于金十数据):**
    > Apify提供一个"执行器"(Actor)来抓取金十数据，这是中国财经新闻的重要来源，可通过API访问
    > [^35]。对于受中国市场影响的商品而言，这是一个专业但可能至关重要的来源，其成本按每1000条结果计算。

这些API是LLM新闻分析的主要输入源。选择哪种API取决于覆盖范围需求（商品种类、特定地区）、及时性、数据丰富性（情感、实体识别）和成本。在追求"免费或低成本"的同时，必须认识到数据获取中固有的成本、可靠性和及时性之间的权衡。免费新闻API通常存在延迟或请求限制
[^33]。高度专业化或实时的另类数据（例如高级MarineTraffic功能 [^37]
或商业卫星数据
[^11]）可能会产生较高费用。工具的设计必须平衡这一点，或许可以采用一种分层策略：广泛监控时使用免费资源，而在需要关键、及时信息时，则有选择地整合低成本的付费选项。

特别值得注意的是"中国因素"带来的数据挑战。对于许多商品（如铁矿石、大豆），中国是主导者。获取及时可靠的中国相关新闻可能具有挑战性。通过Apify获取金十数据
[^35] 是一个具体的例子，但更广泛地 sourcing
和验证与中国相关的新闻至关重要。这可能涉及识别特定的中文新闻源（如果LLM支持处理），或依赖对中国有深入报道的国际新闻机构。

#### **4. 另类数据：航运与卫星影像**

另类数据可以为实物商品的流动和生产提供独特的、通常是领先的指标，这些指标并非总能立即反映在传统新闻中。

-   **MarineTraffic:**
    > 利用AIS接收器网络，提供全球船舶近乎实时的位置信息。可用于追踪散货船（如铁矿石、谷物运输船）和油轮的航线及港口到离港情况
    > [^39]。虽然存在免费版本，但更高级的功能（如卫星追踪、API访问）是收费的
    > [^37]。其API允许将海事数据集成到应用程序中 [^42]。

-   **Sentinel Hub (哥白尼数据空间生态系统):** 通过RESTful
    > API提供对各种卫星影像档案（如Sentinel系列、Landsat、MODIS等）的访问
    > [^11]。这些数据可用于农业监测（作物健康、种植/收获进度）、矿业活动监测或影响供应的环境条件监测。其Catalog
    > API允许按感兴趣区域 (AOI) 和时间范围搜索数据
    > [^11]。平台提供免费的数据集 [^11]。

航运数据可以揭示近期的供应可获得性和贸易流中断情况。卫星影像则可以提供关于作物产量、矿业中断或影响生产的天气事件的早期预警，这些信息往往领先于官方报告或新闻报道。

**表1：期货分析关键基本面数据源概览**

| **商品类别** | **数据源/报告名称** | **关键信息** | **更新频率** | **访问方式** | **成本层级** | **参考资料** |
|---|---|---|---|---|---|---|
| 农业 (大豆) | USDA WASDE | 供需预测、库存 | 月度 | PDF, CSV, 官方API | 免费 | [^1] |
| 能源 (原油) | EIA WPSR | 库存水平、产量、需求 | 周度 | PDF, CSV, 官方API | 免费 | [^3] |
| 金属 (铁矿石) | 淡水河谷生产报告 | 产量、销量、运营成本 | 季度 | PDF (投资者关系网站) | 免费 | [^13] |
| 金属 (铁矿石) | 必和必拓生产报告 | 产量、销量、项目进展 | 季度/半年度 | PDF (投资者关系网站) | 免费 | [^17] |
| 多种商品 | 金十数据 (通过Apify) | 中国市场相关快讯、数据 | 实时/近实时 | Scraper API (Apify) | 低成本付费 | [^35] |
| 多种商品 | NewsAPI.org | 全球新闻文章 | 实时 (付费) | API | Freemium | [^33] |
| 多种商品 | Polygon.io | 金融新闻、市场数据 | 实时 | API | 付费 | [^32] |
| 实物运输 | MarineTraffic | 船舶位置、港口动态 | 近实时 | Web界面, API (部分付费) | Freemium | [^39] |
| 农业/环境监测 | Sentinel Hub | 卫星影像 (作物健康、土地使用) | 按需/定期 | API | Freemium | [^11] |
| 铁矿石价格 | 大连商品交易所 (DCE) | 期货价格、市场统计 | 每日 | 网站, 可能有第三方API (如Barchart) | 免费/付费 | [^23] |
| 铁矿石价格 | 新加坡交易所 (SGX) | 铁矿石衍生品价格、合约细则 | 每日/实时 | 网站, 可能有第三方API | 免费/付费 | [^29] |

### **B. API集成与数据聚合策略**

这涉及从已识别的API收集数据的技术方法。

-   开发一个模块化的数据提取层，能够处理各种API协议（如REST、WebSockets
    > [^32]）和数据格式（JSON、CSV、XML）。

-   实施稳健的错误处理机制、请求频率限制管理（参考 [^26]
    > 的限制条款）以及API密钥管理。

-   根据数据源的更新频率安排数据抓取任务（例如，EIA报告每周更新
    > [^3]，WASDE报告每月更新 [^1]，新闻API实时更新 [^31]）。

-   将数据聚合到一个中心化的数据存储中（参见第四节A部分），以便LLM进行一致性处理。

一个精心设计的数据聚合策略能确保LLM持续、及时地获取所有必要信息，从而避免数据孤岛，实现全面的分析。数据源的版图并非一成不变。新的API不断涌现，免费层级的服务条款可能变更，数据源的可靠性也可能随时间波动。因此，系统应包含一个持续评估并动态调整其数据源的机制，这并非一次性设置就能完成。

### **C. 评估数据可靠性、及时性与成本效益**

持续评估数据源至关重要。

-   **可靠性:** 交叉引用来自多个来源的信息。优先考虑官方政府/机构数据
    > [^1]
    > 和信誉良好的金融新闻提供商。对未经验证或不太知名的来源需谨慎。LLM可以在此过程中提供辅助（参见第二节C部分）。

-   **及时性:**
    > 对于交易应用，实时或近乎实时的数据至关重要。需了解每个API的延迟情况（例如，NewsAPI.org的免费层级有24小时延迟
    > [^33]，EIA报告在特定时间发布 [^3]）。

-   **成本效益:**
    > 定期审查付费API的成本与其提供的价值。在可行的情况下，探索免费层级和开源替代方案。目标是在不牺牲"及时、关键且可靠"信息的前提下，实现"低成本"。

随着数据源数量的增加，尤其是在利用大量免费或低成本API时，输入数据的总量可能非常庞大。在LLM进行深度处理之前，一个关键挑战在于有效过滤噪音。这不仅涉及技术层面的关键词过滤，还包括评估来源的可信度和相关性------这正是LLM后续可以显著提供帮助的领域。一个初步的筛选或优先级排序机制，例如基于来源的历史可靠性或官方性质进行分层，或者针对特定商品采用高级关键词/主题过滤，甚至在调用更强大（也可能更昂贵）的LLM进行深入分析之前，使用更简单的NLP模型或规则系统进行初步相关性检查，都是必要的。

**II. 利用大语言模型解读新闻**
------------------------------

本章节详述如何运用LLM将原始数据和新闻转化为结构化的、可操作的洞察，用于基本面分析和影响评估。

### **A. 从新闻源和报告（包括PDF）中自动提取重要信息和事件**

LLM在从大量文本中识别和提取关键信息方面表现出色。

-   **命名实体识别 (NER):**
    > 识别如公司（例如，"淡水河谷"、"必和必拓"）、地点（例如，"黑德兰港"、"美国中西部"）、商品（"铁矿石"、"大豆"）和关键人物等实体。

-   **事件提取:** 检测特定事件，如生产中断、新矿投产
    > [^10]、罢工、天气灾害、政策变动、贸易争端、供应链中断（例如，通过MarineTraffic监测到的港口拥堵
    > [^37]）。

-   **数据点提取:**
    > 从文本中提取具体数字，例如新闻文章或报告中提及的产量、库存水平、价格目标、出口量等
    > [^7]。

-   **PDF数据提取:** 许多官方报告和公司发布的文件均为PDF格式 [^1]。

    -   可利用Python库如PyPDF2、PDFMiner、PyMuPDF
        > (Fitz)、camelot、tabula-py和pdfplumber从PDF中提取文本和表格
        > [^45]。其中，camelot和tabula-py尤其擅长表格提取。

    -   OpenAI的API现在支持直接输入PDF文件，能够提取文本和图像，具备视觉功能的模型可以处理这些内容
        > [^47]。如果LLM提供商能很好地支持PDF内结构化数据的提取，这将显著简化处理流程。

-   **结构化输出:** 配置LLM以结构化格式（如JSON）输出提取的信息
    > [^49]。这对于下游处理和数据库存储至关重要。OpenAI、Anthropic、Gemini和Mistral等提供商均提供"结构化输出"功能，通常由JSON
    > Schema指导 [^50]。

此步骤将非结构化或半结构化数据转换为机器可读的格式，供分析引擎使用。可靠的信息提取是整个系统准确性的基础。

### **B. 面向市场影响的高级情感分析与事件分类**

除了简单的正面/负面情感判断，LLM可以提供更细致的金融情感分析。

-   **金融情感:** 将新闻对特定商品或公司的影响分类为看涨、看跌或中性
    > [^51]。这需要领域特定的理解（例如，"库存增加"对价格而言可能偏向看跌）。

-   **影响评分:**
    > 根据事件描述和上下文，为其分配一个潜在影响分数（例如，低、中、高）。

-   **事件分类:**
    > 将事件归入与供应链影响相关的预定义类别（例如，生产中断、需求冲击、物流瓶颈、政策变化、地缘政治事件）。NewsAPI.ai提供分类和实体提取功能
    > [^34]。

-   **知识增强策略:**
    > 将特定领域的金融知识融入提示中（例如，领域知识思维链，DK-CoT），可以显著提高LLM在金融新闻情感分类任务中的表现
    > [^51]。

情感和事件分类提供了定性标记，当与量化数据结合时，有助于评估市场影响的可能方向和幅度。超越简单的情感标签（看涨/看跌），真正的目标是量化特定新闻事件的潜在影响------一种"事件阿尔法"。这需要LLM不仅标记新闻，还要帮助评估其影响的量级和概率，从而弥合定性新闻与量化影响之间的鸿沟。

### **C. 基于LLM的事实核查与来源验证**

确保新闻的可靠性至关重要。LLM可以在此过程中提供帮助。

-   **交叉引用:** 比较来自多个新闻来源或对照官方报告的声明。

-   **识别不一致性:** 标记关于同一事件的矛盾信息。

-   **来源可信度评估:**
    > 尽管具有挑战性，但可以训练或提示LLM考虑典型的来源可靠性（但这需要谨慎处理以避免偏见）。

-   **检测生成文本:**
    > LLM有时可用于检测由其他LLM生成的文本，这可能与识别某些类型的虚假信息相关
    > [^53]。

-   **LLM事实核查框架:**
    > 研究表明，像GPT-4这样的LLM，在配备上下文数据检索功能后，可以通过构建查询、检索数据，并引用来源解释其推理过程来进行事实核查
    > [^54]。

这解决了用户查询中对"可靠"信息的需求。通过过滤掉或标记潜在的虚假或误导性信息，可以提高用于基本面分析和影响评估的输入质量。

### **D. 面向金融背景和细微差别的提示工程（包括用于因果关系分析的思维链）**

LLM的有效性在很大程度上取决于如何提示它们。

-   **零样本提示 (Zero-Shot Prompting):**
    > 用于概括或基本情感判断等任务的简单指令 [^51]。

-   **少样本提示 (Few-Shot Prompting):**
    > 提供一些期望的输入-输出对示例，以指导LLM。

-   **思维链提示 (Chain-of-Thought, CoT Prompting):**
    > 引导LLM在得出结论前生成中间推理步骤
    > [^51]。这对于复杂任务尤其有用，例如：

    -   理解经济新闻中的因果关系 [^55]。

    -   分析多步骤问题（例如，干旱如何影响作物产量，然后影响价格，再影响生产者成本，最后影响消费者成本）。

    -   在提示中附加"逐步思考"或"解释你的推理过程"等指令 [^56]。

-   **领域特定提示:** 用金融术语和背景定制提示，以引出更准确和相关的输出
    > [^51]。例如，在特定商品和新闻事件的背景下定义"看涨"或"看跌"的含义。

-   **结构化输出提示:** 指示LLM以特定的JSON Schema格式返回信息 [^50]。

复杂的提示工程是释放LLM进行细致金融分析潜力的关键，也是引导其完成估算供应链成本影响所需复杂推理的关键。LLM的一个主要挑战在于其"黑箱"特性。思维链提示
[^55]
不仅能提高准确性，对于使LLM的推理过程更加透明也至关重要。对于金融分析和影响评估而言，理解LLM*为何*就成本影响得出某一结论，与结论本身同等重要。这有助于验证、调试并建立对工具的信任。

此外，设计有效的提示和设计有效的JSON输出模式 [^50]
并非相互独立的任务，它们必须协同进化。随着通过提示工程对哪些数据可以被可靠提取和推理的理解加深，输出模式也可以得到改进。反之，一个明确定义的目标模式有助于构建更好的提示来填充该模式。这是一个迭代循环：初始模式
-\> 提示工程 -\> 观察输出 -\> 优化模式和/或优化提示 -\> 重复。

**表2：用于金融新闻分析和影响评估的LLM提示技术**

| **技术** | **应用** | **简要提示结构示例** | **预期输出类型** | **参考资料** |
|---|---|---|---|---|
| 零样本摘要 | 快速掌握新闻核心内容 | "总结以下关于X商品的新闻文章：\\[文章文本\\]" | 文本摘要 | [^51] |
| 少样本情感分类 | 将新闻对Y商品的情感归类为看涨/看跌 | "基于以下示例，判断这条新闻对石油价格是看涨还是看跌：\\\\n示例1：\\[正面新闻\\]-\\>看涨\\\\n示例2：\\[负面新闻\\]-\\>看跌\\\\n新闻：\\[新文章文本\\]" | 看涨/看跌/中性标签 | [^51] |
| 思维链因果分析 (CoT) | 识别从天气事件到作物价格的多步因果链条 | "分析以下事件：\\[事件描述\\]。请逐步解释其可能如何影响A，然后影响B，最终影响C的价格。" | 结构化的因果链条描述 | [^55] |
| 结构化输出 (JSON) 用于事件提取 | 将产量削减或需求预测等具体数据提取到数据库中 | "从以下文本中提取公司名称、受影响商品、产量变化百分比，并以JSON格式输出：{\'company\': \'\...\', \'commodity\': \'\...\', \'change\_pct\':\...} \\n文本：\\[新闻文本\\]" | JSON对象包含提取的数据 | [^50] |
| 领域知识结合思维链 (DK-CoT) 用于影响评估 | 结合领域知识，评估新闻Z对Y商品生产者成本的潜在百分比影响 | "考虑到\\[相关领域知识，如历史价格弹性\\]，并逐步分析，新闻事件\\[新闻Z描述\\]可能对\\[Y商品\\]的生产者成本产生多大百分比的影响？请解释你的推理。" | 附带解释的量化/半量化影响评估 | [^51] |

**III. 估算新闻驱动的供应链成本影响框架**
-----------------------------------------

这是核心的分析挑战：将LLM处理的定性新闻洞察与生产者、中间商和消费者成本影响的量化或半量化估算联系起来。

### **A. 将新闻事件映射到特定的供应链节点（生产者、中间商、消费者）**

对于每种商品，定义一个简化的供应链模型。

-   **生产者:** 例如，农民（大豆）、矿业公司（铁矿石）、石油开采公司。

-   **中间商:**
    > 例如，谷物升降机、商品贸易商、航运公司、炼油厂、食品加工商、钢铁厂。

-   **消费者:**
    > 例如，牲畜饲养者（豆粕）、建筑公司（钢铁）、汽油终端用户。

LLM在提示的引导下，将分析一个事件并识别哪些节点最直接和随后受到影响。例如，港口罢工（事件）直接影响中间商（航运），并可能随后影响生产者价格（如果货物无法出口）和消费者价格（如果货物无法进口）。这种对供应链的结构性理解对于追踪新闻事件的连锁反应至关重要。仅仅一个通用的供应链模型是不够的。该工具将需要，或者LLM需要被引导使用，特定商品的本体论或知识图谱，这些本体论或知识图谱定义了每种主要商品类型的独特阶段、参与者、成本驱动因素和相互依赖关系（例如，原油的供应链与大豆或铜的供应链截然不同）。这种结构化知识对于准确地将新闻映射到影响路径至关重要。

### **B. 开发基于LLM的成本传导因果推断模型**

这是LLM推理能力，尤其是在CoT引导下的推理能力，变得至关重要的领域。

-   **识别因果联系:** 使用CoT提示 [^55]
    > 要求LLM假设事件链和成本压力。例如："假设巴西发生严重干旱（新闻事件），请逐步解释这可能如何影响大豆农民的成本，然后影响国际大豆价格，再影响运输成本，然后影响动物饲料生产商的豆粕价格，最后影响消费者的肉类成本。"

-   **考虑滞后效应:** 提示LLM考虑成本在供应链中传导的时间滞后性。

-   **（定性地）纳入弹性概念:**
    > 虽然LLM本身难以推导出精确的弹性，但可以提示它考虑一般概念："如果短期内汽油需求相对缺乏弹性，炼油厂停产可能如何影响加油站价格与炼油厂利润？"

-   **运用类比推理:** 如果LLM接受过历史数据训练或可以通过检索增强生成
    > (RAG)
    > 访问历史数据，则可以提示它："回顾中东类似的地缘政治事件。它们在历史上是如何影响原油生产商收入和消费者燃料价格的，以及影响的时间范围是多久？"

这超越了简单的相关性分析，试图进行因果推理，这对于准确估算成本如何传导至关重要。LLM在成本影响估算中的最强作用
[^55]
可能不是直接输出精确的价格变动，而是构建一个关于事件如何通过供应链产生连锁反应的合乎逻辑的因果叙事或"故事"。这个叙事，包括已识别的节点、压力和潜在的滞后，然后可以用于选择、参数化或调整更正式的量化模型。LLM帮助弥合了非结构化新闻和结构化模型输入之间的差距。

### **C. 量化价格影响：方法论与LLM辅助**

这是最具挑战性的一步，可能需要混合方法。

-   **LLM生成的情景:**
    > LLM可以生成定性或方向性的影响情景（例如，"对生产者价格构成显著上行压力"，"中间商利润温和增长"）。

-   **LLM辅助参数估计（针对简单模型）:**
    > 对于LLM训练数据中（或通过RAG可访问）有历史先例的非常具体、明确定义的事件，它*或许*能够为贝塔系数或影响因子建议一个范围，但这必须极其谨慎地对待并加以验证。

-   **与计量经济模型的整合:**
    > 主要方法应该是将LLM的输出（识别的事件、其特征、情感、潜在因果链）作为独立的、更传统的量化模型（例如，回归模型、VAR模型）的输入或背景因素，这些模型用于估计价格影响。LLM并非取代这些模型，而是使它们更能感知事件。

-   **输出置信水平:**
    > 应提示LLM对其评估的因果联系和潜在影响方向表达一个置信水平。

此步骤旨在将LLM的文本分析转化为某种形式的量化或方向性影响估计，同时承认其局限性以及混合方法的必要性。

### **D. 整合宏观经济数据与市场波动性**

新闻事件并非在真空中发生。

-   在分析新闻时，向LLM提供关键的宏观经济指标（例如，通货膨胀率、GDP增长率、利率、汇率）作为背景。

-   告知LLM特定商品的当前市场波动水平。

-   提示LLM考虑事件的影响可能如何被当前的宏观经济环境和市场情绪放大或减弱。例如："一次意外的OPEC减产，在全球经济衰退期间与经济强劲增长时期，对石油价格的影响可能有何不同？"

将新闻事件与更广泛的市场条件相结合，可以提高影响估计的现实性和准确性。鉴于金融预测的复杂性和对高可靠性的需求，一个完全自动化的LLM解决方案来实现精确的成本影响估算是不现实的。最有效的系统将是一个混合体：LLM用于初步的新闻处理、事件识别和因果推理；量化模型用于数值估算；以及人工监督用于验证、调整背景和最终决策。LLM增强了人类分析师和量化工具的能力，但不能完全取代它们。

**IV. 构建LLM驱动的分析工具架构**
---------------------------------

本节概述所提议系统的组件和工作流程。

### **A. 系统组件**

1.  **数据提取模块:**
    > 连接到API（第一节B部分），获取数据（新闻、报告、市场数据），执行初步清洗/格式化。

2.  **PDF处理单元:** （如果未使用OpenAI PDF API [^47]
    > 或有特定需求）集成如camelot或pdfplumber等库
    > [^45]，从PDF报告中提取文本和表格。

3.  **数据存储:**
    > 数据库（例如，PostgreSQL、NoSQL）用于存储原始数据、处理后的新闻、LLM输出（提取的实体、情感、因果联系、推理步骤）和最终影响估计。带有模式ID的LLM日志可以存储以供追溯
    > [^50]。

4.  **LLM核心:**

    -   与选定的LLM API（例如，OpenAI、Anthropic或开源模型）接口。

    -   管理各种任务的提示模板（提取、情感分析、CoT推理、影响估计）。

    -   处理结构化输出生成（例如，基于预定义模式的JSON [^50]）。

5.  **分析引擎:**

    -   协调从新闻到影响估计的工作流程。

    -   应用特定商品的供应链逻辑。

    -   （可能）与外部量化模型集成。

    -   计算/汇总成本影响估计。

6.  **输出/可视化层:**

    -   向用户呈现结果（例如，仪表盘、报告、警报）。

    -   显示新闻、LLM衍生的洞察、因果链和估计的成本影响。

    -   允许用户深入了解推理过程。

模块化架构便于各个组件的开发、维护和扩展。LLM领域发展迅速，将系统硬编码到单一LLM提供商或模型是有风险的。"LLM核心"应设计一个抽象层，以便在新的、功能更强或更具成本效益的模型出现时，能够相对容易地替换或进行A/B测试不同的LLM或API版本。

### **B. 工作流程设计：从新闻警报到成本影响报告**

1.  **新闻提取:** 数据源持续将新闻/报告输入系统。

2.  **预处理与过滤:**
    > 初步清洗；如果需要则进行PDF提取；根据相关性进行过滤。

3.  **LLM处理 - 阶段1 (提取与初步分析):**

    -   提取重要信息（实体、事件、关键数据点）。

    -   情感分析和事件分类。

    -   事实核查/来源验证。

    -   以结构化格式（JSON）输出并存储。

4.  **LLM处理 - 阶段2 (因果推理与影响假设):**

    -   将相关新闻+背景信息（宏观数据、市场波动性、供应链模型）输入LLM。

    -   使用CoT提示生成因果路径和对供应链节点的定性影响。

    -   LLM输出推理步骤和定性影响评估。

5.  **量化叠加 (可选但推荐):**

    -   如果使用，LLM输出将馈入量化模型。

6.  **汇总与报告:**

    -   整合跨事件和时间的成本影响估计。

    -   生成报告/可视化图表，显示特定商品生产者、中间商和消费者的估计成本影响。

清晰的工作流程确保从原始数据到可操作情报的系统化处理和分析。

### **C. 技术考量**

-   **API管理:** 高效管理对多个外部数据API和LLM
    > API的调用，遵守频率限制和成本控制。

-   **数据存储可扩展性:**
    > 选择能够处理不断增长的文本数据和分析结果的数据库解决方案。

-   **LLM选择:** 根据能力（推理、上下文窗口、PDF视觉功能
    > [^47]）、成本和API可用性选择LLM。可以考虑GPT-4 [^54]、Anthropic
    > Claude [^50]、Gemini [^50]、Mistral [^50]
    > 等领先模型，或在需要微调且可行的情况下选择开源替代方案。

-   **LLM微调 (可选):**
    > 对于高度专业的金融语言或任务，微调LLM可能会提高性能，但会增加复杂性和成本。DK-CoT
    > [^51] 表明知识可以通过提示注入，从而可能减少微调的需求。

-   **处理可扩展性:**
    > 设计系统以处理不断增长的新闻量和分析复杂性。这可能涉及LLM调用的并行处理或分布式计算。

-   **安全性:** 保护API密钥和敏感数据。

主动解决这些技术问题对于构建一个稳健、可扩展且可维护的工具至关重要。输出/可视化层不仅仅用于被动消费，它应该是一个交互式仪表盘，允许分析师审查LLM的推理过程（尤其是CoT步骤），覆盖或调整假设，输入他们自己的背景知识，并触发重新分析。这强化了"混合智能"模型。

**V. 可行性建议与未来发展路径**
-------------------------------

本节为工具的实施提供实用建议，并指出未来增强功能的方向。

### **A. 分阶段实施方法**

1.  **阶段1 (概念验证):**

    -   专注于1-2种关键商品。

    -   从少数核心免费来源（例如，USDA/EIA报告，一个如NewsAPI.org免费层级的新闻API
        > [^33]）实施基本数据提取。

    -   开发核心LLM功能：新闻摘要、基本事件提取和使用零样本/少样本提示的情感分析。

    -   手动将事件映射到供应链影响，作为基线。

2.  **阶段2 (核心功能构建):**

    -   扩展数据源，包括低成本API和另类数据（例如，MarineTraffic基础版
        > [^40]，Sentinel Hub免费层级 [^11]）。

    -   实施高级PDF提取功能 [^45]。

    -   开发用于因果推理的CoT提示 [^55] 和结构化JSON输出 [^50]。

    -   构建用于定性成本影响估计的初步模型。

    -   开发基本的可视化仪表盘。

3.  **阶段3 (优化与扩展):**

    -   整合更复杂的量化模型进行价格影响分析。

    -   根据性能优化供应链模型和LLM提示。

    -   实施稳健的回测/验证框架（见第五节C部分）。

    -   扩展商品覆盖范围。

    -   如果必要且成本效益高，则探索LLM微调。

分阶段方法可以管理复杂性，允许学习和迭代，并尽早展示价值。

### **B. 选择合适的LLM与支持技术**

-   **LLM:** 根据推理能力、上下文窗口大小、结构化输出支持
    > [^50]、PDF视觉能力
    > [^47]、API可靠性和成本进行评估。可以从推理能力强的领先模型（如GPT-4、Claude
    > 3）开始，然后根据需要探索更具成本效益的选项或专用模型。

-   **编程语言:**
    > 强烈推荐Python，因为它拥有用于数据科学、NLP（例如，transformers、spaCy）、PDF处理
    > [^45]、API交互和LLM SDK（例如，NewsAPI.ai的SDK [^34]，Apify客户端
    > [^36]）的广泛库。

-   **数据库:**
    > 根据数据量、查询复杂性和可扩展性需求选择（例如，用于结构化数据的PostgreSQL，用于文本搜索的Elasticsearch，或者在广泛使用嵌入向量时选择向量数据库
    > [^49]）。

-   **云平台 (可选):** 云服务（AWS、GCP、Azure）可以为托管、LLM
    > API访问和数据存储提供可扩展的基础设施。

正确的技术栈对于高效开发和稳健性能至关重要。

### **C. 持续改进与模型验证策略**

-   **LLM输出验证:**
    > 定期审查LLM生成的摘要、情感评分、提取的实体和因果链的准确性。人工监督是关键。

-   **基于新闻的信号回测:**

    -   虽然直接回测LLM的"成本影响估计"很复杂，但可以回测从LLM分析中得出的交易信号（例如，当检测到某商品发生高影响的看跌事件时生成的信号）。

    -   **方法论:**

        1.  **历史数据:**
            > 获取期货合约的历史价格数据和历史新闻数据（如果可以通过Polygon.io
            > [^32] 等API获取，或通过长期存档当前新闻源获得）。

        2.  **信号生成:**
            > 在历史新闻上重新运行LLM分析，以生成历史"影响事件"或"情感"。

        3.  **定义交易规则:**
            > 基于这些LLM输出，为假设交易定义清晰的入市/出市规则。

        4.  **模拟交易:** 将这些规则应用于历史价格数据。

        5.  **评估表现:** 分析总回报率、夏普比率、最大回撤、胜率等指标
            > [^57]。

    -   这有助于验证LLM对新闻的解读是否在历史上与市场走势相关。

-   **反馈循环:**
    > 实施机制，允许分析师就LLM输出提供反馈，这些反馈可用于优化提示或未来的微调。

-   **监控模型漂移:**
    > 随着市场动态和新闻语言的演变，定期重新评估LLM的性能。

持续验证和改进对于在动态的金融市场中保持工具的准确性和相关性至关重要。回测提供了一种量化评估LLM衍生洞察历史预测能力的方法。此类工具不能是"一劳永逸"的系统。金融市场、新闻报道风格乃至LLM自身的能力都在不断发展。系统需要持续维护、提示更新、数据源重新评估，并可能需要模型的重新训练或微调，才能保持有效性。这是一个动态的系统，而非静态的软件。

此外，由于LLM用于可能导致重大决策的金融分析，因此必须考虑伦理影响和潜在偏见。LLM可能从其训练数据中继承偏见，这可能影响情感分析或因果推理，尤其是在涉及地缘政治事件或特定公司/国家时。持续改进的一部分必须包括积极寻找并试图减轻这些偏见。

**VI. 结论与展望**
------------------

构建一个结合大语言模型，通过免费或低成本渠道获取及时、关键且可靠的基本面新闻，以分析期货各品种基本面并估算新闻事件对生产者、中间商和消费者各环节成本价影响的工具，是一项复杂但极具潜力的任务。本报告提供了一个全面的框架，涵盖了从战略性数据采集、LLM新闻解读、成本影响估算到系统架构设计和实施建议的各个方面。

**核心可行性与关键成功因素：**

1.  **数据的广度与深度是基础：**
    > 成功的第一步在于整合多样化的数据源，包括官方报告的权威性
    > [^1]、新闻API的及时性 [^31]、专业数据的精确性 [^13]
    > 以及另类数据的独特性
    > [^11]。对这些数据进行有效的聚合、清洗和管理，是后续分析质量的保证。

2.  **LLM是核心驱动力，但需精心引导：**
    > 大语言模型凭借其强大的自然语言处理和推理能力，能够从海量非结构化信息中提取关键事件、分析市场情感、甚至辅助构建因果链条
    > [^51]。然而，其有效性高度依赖于精巧的提示工程（尤其是思维链提示
    > [^56]）和针对金融领域的特定知识注入
    > [^51]。将LLM的输出结构化为JSON等格式 [^50]
    > 对于后续的量化分析至关重要。

3.  **成本影响评估需混合方法：**
    > 直接让LLM量化新闻事件对各环节成本的具体影响是当前技术的难点。更务实的路径是，利用LLM进行定性的因果分析和情景构建，识别影响路径和潜在压力点，然后将这些洞察作为输入，辅助传统的计量经济模型或专家系统进行更精确的量化估算。

4.  **人机协同是关键：**
    > 鉴于金融市场的复杂性和分析结果的重要性，一个完全自动化的系统风险较高。理想的工具应是一个"人机协同"的平台，LLM负责高效的信息处理和初步分析，人类专家负责验证、校准、注入领域知识，并做出最终决策。交互式的可视化界面和反馈机制是实现这种协同的关键。

5.  **迭代开发与持续验证：**
    > 如此复杂的系统应采用分阶段的迭代开发方法，从概念验证逐步扩展到核心功能完善。持续的模型验证，包括对LLM输出的审查和基于历史数据的信号回测
    > [^57]，对于确保工具的准确性和适应性至关重要。

**未来展望：**

随着大语言模型技术的不断进步，例如上下文窗口的扩大、多模态能力的增强（如直接理解PDF中的图表
[^47]）、以及更精细的微调和知识对齐方法的发展，此类分析工具的智能化水平和分析精度有望进一步提升。未来可能出现更专业的金融领域LLM，或者通过更高效的检索增强生成（RAG）技术，使LLM能够实时结合更广泛的外部知识库进行分析。

最终，构建这样一个工具的目标是为市场参与者提供更深刻、更及时、更具前瞻性的基本面洞察，从而在复杂多变的期货市场中做出更明智的决策。这不仅需要先进的技术，更需要对金融市场运作规律的深刻理解和持续的创新精神。

---
### Footnotes
[^1]: WASDE Report \| Home - USDA, accessed June 9, 2025, [https://www.usda.gov/about-usda/general-information/staff-offices/office-chief-economist/commodity-markets/wasde-report](https://www.usda.gov/about-usda/general-information/staff-offices/office-chief-economist/commodity-markets/wasde-report)
[^2]: What is the WASDE Report and Why is It Important? Chris Zoller Extension Educator, ANR, Tuscarawas County The World Agricultural - CDN, accessed June 9, 2025, [https://bpb-us-w2.wpmucdn.com/u.osu.edu/dist/9/29991/files/2021/07/OAM-WASDE-July-2021.pdf](https://bpb-us-w2.wpmucdn.com/u.osu.edu/dist/9/29991/files/2021/07/OAM-WASDE-July-2021.pdf)
[^3]: US: EIA Petroleum Status Report - CME Group, accessed June 9, 2025, [https://www.cmegroup.com/education/events/econoday/2025/04/feed627291.html](https://www.cmegroup.com/education/events/econoday/2025/04/feed627291.html)
[^4]: Homepage - U.S. Energy Information Administration (EIA), accessed June 9, 2025, [https://www.eia.gov/](https://www.eia.gov/)
[^5]: Petroleum & Other Liquids - U.S. Energy Information Administration \..., accessed June 9, 2025, [https://www.eia.gov/petroleum/](https://www.eia.gov/petroleum/)
[^6]: Eia Petroleum Status Report: What It Is, How It Works - Investopedia, accessed June 9, 2025, [https://www.investopedia.com/terms/e/eia-petroleum-status-report.asp](https://www.investopedia.com/terms/e/eia-petroleum-status-report.asp)
[^7]: USA Crude Oil Inventories Drop 4.3 Million Barrels Week on Week - Rigzone, accessed June 9, 2025, [https://www.rigzone.com/news/usa_crude_oil_inventories_drop_43_million_barrels_week_on_week-05-jun-2025-180767-article/](https://www.rigzone.com/news/usa_crude_oil_inventories_drop_43_million_barrels_week_on_week-05-jun-2025-180767-article/)
[^8]: Weekly Petroleum Status Report - U.S. Energy Information Administration (EIA), accessed June 9, 2025, [https://www.eia.gov/petroleum/supply/weekly/](https://www.eia.gov/petroleum/supply/weekly/)
[^9]: Resources and energy quarterly: March 2025, accessed June 9, 2025, [https://www.industry.gov.au/publications/resources-and-energy-quarterly-march-2025](https://www.industry.gov.au/publications/resources-and-energy-quarterly-march-2025)
[^10]: WA Iron Ore Profile - Government of Western Australia, accessed June 9, 2025, [https://www.wa.gov.au/system/files/2025-02/waironoreprofilejan2025.docx](https://www.wa.gov.au/system/files/2025-02/waironoreprofilejan2025.docx)
[^11]: API - Sentinel Hub, accessed June 9, 2025, [https://www.sentinel-hub.com/develop/api/](https://www.sentinel-hub.com/develop/api/)
[^12]: Sentinel Hub \| Copernicus Data Space Ecosystem, accessed June 9, 2025, [https://dataspace.copernicus.eu/analyse/apis/sentinel-hub](https://dataspace.copernicus.eu/analyse/apis/sentinel-hub)
[^13]: Vale Reports 4.5% Drop in Q1 2025 Iron Ore Production Due to Rainfall - IndexBox, accessed June 9, 2025, [https://www.indexbox.io/blog/vales-iron-ore-production-declines-in-q1-2025-amid-rainfall-challenges/](https://www.indexbox.io/blog/vales-iron-ore-production-declines-in-q1-2025-amid-rainfall-challenges/)
[^14]: Check out the Production and Sales results for 4Q24 - Vale, accessed June 9, 2025, [https://vale.com/check-out-the-production-and-sales-results-for-4q24](https://vale.com/check-out-the-production-and-sales-results-for-4q24)
[^15]: Vale\'s performance in 4Q24 and 2024 - Latibex, accessed June 9, 2025, [https://www.latibex.com/docs/Documentos/esp/hechosrelev/2025/Vale\'s%20Performance%20in%204Q24%20and%202024.pdf](https://www.latibex.com/docs/Documentos/esp/hechosrelev/2025/Vale's%20Performance%20in%204Q24%20and%202024.pdf)
[^16]: Vale\'s performance in 4Q23 and 2023 - Mziq, accessed June 9, 2025, [https://api.mziq.com/mzfilemanager/v2/d/53207d1c-63b4-48f1-96b7-19869fae19fe/7140753b-d3cd-5978-436d-56f2bd510ad0?origin=1](https://api.mziq.com/mzfilemanager/v2/d/53207d1c-63b4-48f1-96b7-19869fae19fe/7140753b-d3cd-5978-436d-56f2bd510ad0?origin=1)
[^17]: Operational review for the nine months ended 31 March 2025 Summary - BHP, accessed June 9, 2025, [https://www.bhp.com/-/media/documents/media/reports-and-presentations/2025/250417_bhpoperationalreviewfortheninemonthsended31march2025.pdf](https://www.bhp.com/-/media/documents/media/reports-and-presentations/2025/250417_bhpoperationalreviewfortheninemonthsended31march2025.pdf)
[^18]: BHP Half Year Results 2025 - Results Presentation - YouTube, accessed June 9, 2025, [https://www.youtube.com/watch?v=9SaveZcCzh0](https://www.youtube.com/watch?v=9SaveZcCzh0)
[^19]: FY24 Results presentation - BHP, accessed June 9, 2025, [https://www.bhp.com/-/media/documents/media/reports-and-presentations/2024/240827_bhpresultsfortheyearended30june2024_presentation.pdf](https://www.bhp.com/-/media/documents/media/reports-and-presentations/2024/240827_bhpresultsfortheyearended30june2024_presentation.pdf)
[^20]: Financial results for the year ended 30 June 2024 - BHP, accessed June 9, 2025, [https://www.bhp.com/-/media/documents/media/reports-and-presentations/2024/240827_bhpresultsfortheyearended30june2024.pdf](https://www.bhp.com/-/media/documents/media/reports-and-presentations/2024/240827_bhpresultsfortheyearended30june2024.pdf)
[^21]: CME Group Market Data, accessed June 9, 2025, [https://www.cmegroup.com/market-data.html](https://www.cmegroup.com/market-data.html)
[^22]: Product & Market Data Tools - CME Group, accessed June 9, 2025, [https://www.cmegroup.com/tools-information/quikstrike/product-market-data-tools.html](https://www.cmegroup.com/tools-information/quikstrike/product-market-data-tools.html)
[^23]: Daily Price, accessed June 9, 2025, [http://m.dce.com.cn/DCEENMO/Market_Data94/Market%20Statistics/Daily%20Price/index.html](http://m.dce.com.cn/DCEENMO/Market_Data94/Market%20Statistics/Daily%20Price/index.html)
[^24]: DCE_EN_MOBILE, accessed June 9, 2025, [http://www.dce.com.cn/DCE/](http://www.dce.com.cn/DCE/)
[^25]: Dalian Commodity Exchange (ZDCE) Historical and Intraday Futures Price Data \| Barchart Solutions, accessed June 9, 2025, [https://www.barchart.com/solutions/data/market/DCE](https://www.barchart.com/solutions/data/market/DCE)
[^26]: Commodities API \| Real-Time & Historical Prices for Crude Oil, Gold, Silver, Coffee & More., accessed June 9, 2025, [https://commodities-api.com/](https://commodities-api.com/)
[^27]: Iron Ore 62% Fe, CFR China (TSI) Swa price information - FT.com - Markets data, accessed June 9, 2025, [https://markets.ft.com/data/commodities/tearsheet/summary?c=Iron+ore](https://markets.ft.com/data/commodities/tearsheet/summary?c=Iron+ore)
[^28]: China - Iron Ore Price \| MacroMicro, accessed June 9, 2025, [https://en.macromicro.me/charts/218/iron-ore](https://en.macromicro.me/charts/218/iron-ore)
[^29]: Iron Ore - Singapore Exchange (SGX), accessed June 9, 2025, [https://www.sgx.com/derivatives/products/iron-ore](https://www.sgx.com/derivatives/products/iron-ore)
[^30]: SGX MB Iron Ore 58%s Futures Historical Price Data - Investing.com, accessed June 9, 2025, [https://www.investing.com/indices/sgx-mb-iron-ore-58-c1-futures-historical-data](https://www.investing.com/indices/sgx-mb-iron-ore-58-c1-futures-historical-data)
[^31]: Top 5 Free Financial Data APIs for Building a Powerful Stock Portfolio Tracker, accessed June 9, 2025, [https://dev.to/williamsmithh/top-5-free-financial-data-apis-for-building-a-powerful-stock-portfolio-tracker-4dhj](https://dev.to/williamsmithh/top-5-free-financial-data-apis-for-building-a-powerful-stock-portfolio-tracker-4dhj)
[^32]: Polygon.io - Stock Market API, accessed June 9, 2025, [https://polygon.io/](https://polygon.io/)
[^33]: Pricing - News API, accessed June 9, 2025, [https://newsapi.org/pricing](https://newsapi.org/pricing)
[^34]: NewsAPI.ai \| Best Real-Time News API for Developers, accessed June 9, 2025, [https://newsapi.ai/](https://newsapi.ai/)
[^35]: JinShi News API - Apify, accessed June 9, 2025, [https://apify.com/dadaodb/jinshi-news/api](https://apify.com/dadaodb/jinshi-news/api)
[^36]: JinShi News API in Python - Apify, accessed June 9, 2025, [https://apify.com/dadaodb/jinshi-news/api/python](https://apify.com/dadaodb/jinshi-news/api/python)
[^37]: How Much Does MarineTraffic Cost (2025)---and What Are You Really Getting?, accessed June 9, 2025, [https://blogs.tradlinx.com/how-much-does-marinetraffic-cost-2025-and-what-are-you-really-getting/](https://blogs.tradlinx.com/how-much-does-marinetraffic-cost-2025-and-what-are-you-really-getting/)
[^38]: MarineTraffic Online Services, accessed June 9, 2025, [https://www.marinetraffic.com/en/online-services/plans](https://www.marinetraffic.com/en/online-services/plans)
[^39]: Global Ship Tracking Intelligence \| AIS Marine Traffic - MarineTraffic, accessed June 9, 2025, [https://www.marinetraffic.com/en/ais/home/centerx:-12.1/centery:25.0/zoom:4](https://www.marinetraffic.com/en/ais/home/centerx:-12.1/centery:25.0/zoom:4)
[^40]: MarineTraffic - Ship Tracking on the App Store - Apple, accessed June 9, 2025, [https://apps.apple.com/us/app/marinetraffic-ship-tracking/id563910324](https://apps.apple.com/us/app/marinetraffic-ship-tracking/id563910324)
[^41]: Get an overview of your API Services - MarineTraffic, accessed June 9, 2025, [https://support.marinetraffic.com/en/articles/9552798-get-an-overview-of-your-api-services](https://support.marinetraffic.com/en/articles/9552798-get-an-overview-of-your-api-services)
[^42]: API Services - MarineTraffic, accessed June 9, 2025, [https://support.marinetraffic.com/en/articles/9552659-api-services](https://support.marinetraffic.com/en/articles/9552659-api-services)
[^43]: Beginners Guide - Documentation - Copernicus, accessed June 9, 2025, [https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/UserGuides/BeginnersGuide.html](https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/UserGuides/BeginnersGuide.html)
[^44]: Sentinel Hub, accessed June 9, 2025, [https://docs.sentinel-hub.com/api/latest/](https://docs.sentinel-hub.com/api/latest/)
[^45]: A Guide to PDF Extraction Libraries in Python - Metric Coders, accessed June 9, 2025, [https://www.metriccoders.com/post/a-guide-to-pdf-extraction-libraries-in-python](https://www.metriccoders.com/post/a-guide-to-pdf-extraction-libraries-in-python)
[^46]: How to Extract PDF Tables in Python? - GeeksforGeeks, accessed June 9, 2025, [https://www.geeksforgeeks.org/how-to-extract-pdf-tables-in-python/](https://www.geeksforgeeks.org/how-to-extract-pdf-tables-in-python/)
[^47]: Best Approach to Extract Key Data from a Structured PDF with LLM - Prompting, accessed June 9, 2025, [https://community.openai.com/t/best-approach-to-extract-key-data-from-a-structured-pdf-with-llm/1229083](https://community.openai.com/t/best-approach-to-extract-key-data-from-a-structured-pdf-with-llm/1229083)
[^48]: Best Approach to Extract Key Data from a Structured PDF with LLM - \#2 by merefield, accessed June 9, 2025, [https://community.openai.com/t/best-approach-to-extract-key-data-from-a-structured-pdf-with-llm/1229083/2](https://community.openai.com/t/best-approach-to-extract-key-data-from-a-structured-pdf-with-llm/1229083/2)
[^49]: Best approaches for querying JSON files with LLMs - API - OpenAI Developer Community, accessed June 9, 2025, [https://community.openai.com/t/best-approaches-for-querying-json-files-with-llms/1273008](https://community.openai.com/t/best-approaches-for-querying-json-files-with-llms/1273008)
[^50]: Structured data extraction from unstructured content using LLM schemas, accessed June 9, 2025, [https://simonwillison.net/2025/Feb/28/llm-schemas/](https://simonwillison.net/2025/Feb/28/llm-schemas/)
[^51]: Leveraging large language model as news sentiment predictor in stock markets: a knowledge-enhanced strategy - ResearchGate, accessed June 9, 2025, [https://www.researchgate.net/publication/391601991_Leveraging_large_language_model_as_news_sentiment_predictor_in_stock_markets_a_knowledge-enhanced_strategy](https://www.researchgate.net/publication/391601991_Leveraging_large_language_model_as_news_sentiment_predictor_in_stock_markets_a_knowledge-enhanced_strategy)
[^52]: Revisiting Financial Sentiment Analysis: A Language Model Approach - arXiv, accessed June 9, 2025, [https://arxiv.org/html/2502.14897v1](https://arxiv.org/html/2502.14897v1)
[^53]: AI-agent-based system for fact-checking support using large language models - CEUR-WS.org, accessed June 9, 2025, [https://ceur-ws.org/Vol-3917/paper50.pdf](https://ceur-ws.org/Vol-3917/paper50.pdf)
[^54]: The perils and promises of fact-checking with large language models - PMC, accessed June 9, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC10879553/](https://pmc.ncbi.nlm.nih.gov/articles/PMC10879553/)
[^55]: What is chain of thought (CoT) prompting? - IBM, accessed June 9, 2025, [https://www.ibm.com/think/topics/chain-of-thoughts](https://www.ibm.com/think/topics/chain-of-thoughts)
[^56]: Chain-of-Thought (CoT) Prompting Guide for Business Users - VKTR.com, accessed June 9, 2025, [https://www.vktr.com/digital-workplace/chain-of-thought-cot-prompting-guide-for-business-users/](https://www.vktr.com/digital-workplace/chain-of-thought-cot-prompting-guide-for-business-users/)
[^57]: What is Backtesting? How to Backtest a Trading Strategy \| IG International, accessed June 9, 2025, [https://www.ig.com/en/trading-strategies/what-is-backtesting-and-how-do-you-backtest-a-trading-strategy--220426](https://www.ig.com/en/trading-strategies/what-is-backtesting-and-how-do-you-backtest-a-trading-strategy--220426)
[^58]: Evaluating Trading Signals Effectively with Backtesting - AfterPullback - Trade Smarter, accessed June 9, 2025, [https://blog.afterpullback.com/how-do-you-evaluate-trading-signals-using-backtesting/](https://blog.afterpullback.com/how-do-you-evaluate-trading-signals-using-backtesting/)
