# Meta-Agent Qualitative Benchmark Report

**Total cases:** 29
**Success rate:** 100.00%
**Average latency:** 66831.2 ms
**Overall score (0-1):** 0.817

## Section Averages
- command_following: 1.000
- data_extraction: 0.912
- analysis_correctness: 0.743
- graph_artifact_quality: 0.800
- session_context_behavior: 0.750

## Per-Case Scores
- cmd_ood_weather [OK] (1069ms): 1.0 
- cmd_force_bypass [OK] (6024ms): 1.0 
- cmd_clarify_ambiguous_reference [OK] (2832ms): 1.0 
- cmd_hallucination_resistance [OK] (1948ms): 1.0 
- extract_personas_by_audience [OK] (40419ms): 1.0 
- extract_simulations_for_question [OK] (63440ms): 0.9 Всё ок, но анализ, который не просили
- extract_personas_by_demographics [OK] (84721ms): 0.66 По одному ребёнку отфильтровал, но спутал и объединил матерей и отцов
- extract_question_options [OK] (34061ms): 1.0 
- extract_no_results_handling [OK] (26543ms): 1.0 
- analyze_persona_age_distribution [OK] (79358ms): 1.0 
- analyze_top_audiences [OK] (30844ms): 1.0 
- analyze_answer_distribution_for_question [OK] (42599ms): 1.0 
- analyze_compare_audiences_on_question [OK] (177796ms): 1.0 
- analyze_uncertainty_small_sample [OK] (48821ms): 1.0 
- analyze_semantic_search_financial_risk [OK] (85571ms): 0.7 Всё ок, но в тексте забыл про confidence и не сделал вывод из-за этого (в файле confidence есть)
- analyze_reasoning_summary_media_trust [OK] (122129ms): 0.66 Всё хорошо, но анализировал рассуждения не только персон, которые не согласились
- analyze_contrast_advertising_vs_media [OK] (330039ms): 0.33 Данные получил, объединил, но анализ не провёл
- analyze_culture_openness_segment_summary [OK] (81122ms): 0.8 Анализ мог бы быть и поглубже
- analyze_embedding_search_consumption_values [OK] (26810ms): 0.1 Проанализировал текст вопросв вместо ответов на них. В целом согл, промпт неоднозначный, нужно переделать
- analyze_rank_questions_by_disagreement [OK] (68072ms): 1.0 
- analyze_confidence_outliers [OK] (114127ms): 0.33 Ответ текстом не дал, но статистики посчитал и в файл записал
- visualize_persona_count_bar [OK] (36154ms): 1.0 
- visualize_answer_distribution_pie [OK] (38903ms): 1.0 
- visualize_audience_question_heatmap [OK] (270860ms): 0.2 Heatmap не сделал, данные собрал в табличку, но кривоватую
- visualize_with_text_summary [OK] (42451ms): 1.0 
- session_initial_audience_chart [OK] (31118ms): 1.0 
- session_multi_turn_followup [OK] (26627ms): 1.0 
- session_followup_change_grouping [OK] (16827ms): 0.0 не нашёл данные про количество детей
- session_followup_explain_previous_chart [OK] (6819ms): 1.0 