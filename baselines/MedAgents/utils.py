from prompt_utils import *
from data_utils import *
import time


def fully_decode(qid, realqid, question, options, gold_answer, handler, args, dataobj):
    start_time = time.time()

    question_domains, options_domains, question_analyses, option_analyses, syn_report, output = "", "", "", "", "", ""
    vote_history, revision_history, syn_repo_history = [], [], []
    total_prompt_tokens = 0
    total_completion_tokens = 0

    if args.method == "base_direct":
        direct_prompt = get_direct_prompt(question, options)
        output, usage = handler.get_output_multiagent(user_input=direct_prompt, temperature=0, system_role="")
        total_prompt_tokens += usage.prompt_tokens
        total_completion_tokens += usage.completion_tokens
        ans, output = cleansing_final_output(output)
    elif args.method == "base_cot":
        cot_prompt = get_cot_prompt(question, options)
        output, usage = handler.get_output_multiagent(user_input=cot_prompt, temperature=0, system_role="")
        total_prompt_tokens += usage.prompt_tokens
        total_completion_tokens += usage.completion_tokens
        ans, output = cleansing_final_output(output)
    else:
        # get question domains
        question_classifier, prompt_get_question_domain = get_question_domains_prompt(question)
        raw_question_domain, usage = handler.get_output_multiagent(user_input=prompt_get_question_domain, temperature=0, system_role=question_classifier)
        total_prompt_tokens += usage.prompt_tokens
        total_completion_tokens += usage.completion_tokens
        if raw_question_domain == "ERROR.":
            raw_question_domain  = "Medical Field: " + " | ".join(["General Medicine" for _ in range(NUM_QD)])
        question_domains = [domain.strip() for domain in raw_question_domain.split("Field:")[-1].strip().split("|")]

        # get option domains
        options_classifier, prompt_get_options_domain = get_options_domains_prompt(question, options)
        raw_option_domain, usage = handler.get_output_multiagent(user_input=prompt_get_options_domain, temperature=0, system_role=options_classifier)
        total_prompt_tokens += usage.prompt_tokens
        total_completion_tokens += usage.completion_tokens
        if raw_option_domain == "ERROR.":
            raw_option_domain  = "Medical Field: " + " | ".join(["General Medicine" for _ in range(NUM_OD)])
        options_domains = [domain.strip() for domain in raw_option_domain.split("Field:")[-1].strip().split("|")]

        # get question analysis
        tmp_question_analysis = []
        for _domain in question_domains:
            question_analyzer, prompt_get_question_analysis = get_question_analysis_prompt(question, _domain)
            raw_question_analysis, usage = handler.get_output_multiagent(user_input=prompt_get_question_analysis, temperature=0, system_role=question_analyzer)
            total_prompt_tokens += usage.prompt_tokens
            total_completion_tokens += usage.completion_tokens
            tmp_question_analysis.append(raw_question_analysis)
        question_analyses = cleansing_analysis(tmp_question_analysis, question_domains, 'question')

        # get option analysis
        tmp_option_analysis = []
        for _domain in options_domains:
            option_analyzer, prompt_get_options_analyses = get_options_analysis_prompt(question, options, _domain, question_analyses)
            raw_option_analysis, usage = handler.get_output_multiagent(user_input=prompt_get_options_analyses, temperature=0, system_role=option_analyzer)
            total_prompt_tokens += usage.prompt_tokens
            total_completion_tokens += usage.completion_tokens
            tmp_option_analysis.append(raw_option_analysis)
        option_analyses = cleansing_analysis(tmp_option_analysis, options_domains, 'option')

        if args.method == "anal_only":
            answer_prompt = get_final_answer_prompt_analonly(question, options, question_analyses, option_analyses)
            output, usage = handler.get_output_multiagent(user_input=answer_prompt, temperature=0, system_role="")
            total_prompt_tokens += usage.prompt_tokens
            total_completion_tokens += usage.completion_tokens
            ans, output = cleansing_final_output(output)
        else:
            # get synthesized report
            q_analyses_text = transform_dict2text(question_analyses, "question", question)
            o_analyses_text = transform_dict2text(option_analyses, "options", options)
            synthesizer, prompt_get_synthesized_report = get_synthesized_report_prompt(q_analyses_text, o_analyses_text)
            raw_synthesized_report, usage = handler.get_output_multiagent(user_input=prompt_get_synthesized_report, temperature=0, system_role=synthesizer)
            total_prompt_tokens += usage.prompt_tokens
            total_completion_tokens += usage.completion_tokens
            syn_report = cleansing_syn_report(question, options, raw_synthesized_report)

            if args.method == "syn_only":
                # final answer derivation
                answer_prompt = get_final_answer_prompt_wsyn(syn_report)
                output, usage = handler.get_output_multiagent(user_input=answer_prompt, temperature=0, system_role="")
                total_prompt_tokens += usage.prompt_tokens
                total_completion_tokens += usage.completion_tokens
                ans, output = cleansing_final_output(output)
            elif args.method == "syn_verif":
                all_domains = question_domains + options_domains

                syn_repo_history = [syn_report]
            
                hasno_flag = True   # default value: in order to get into the while loop
                num_try = 0

                while num_try < args.max_attempt_vote and hasno_flag:
                    domain_opinions = {}    # 'domain' : 'yes' / 'no'
                    revision_advice = {}
                    num_try += 1
                    hasno_flag = False
                    # hold a meeting for all domain experts to vote and gather advice if they do not agree
                    for domain in all_domains:
                        voter, cons_prompt = get_consensus_prompt(domain, syn_report)
                        raw_domain_opi, usage = handler.get_output_multiagent(user_input=cons_prompt, temperature=0, system_role=voter)
                        total_prompt_tokens += usage.prompt_tokens
                        total_completion_tokens += usage.completion_tokens
                        domain_opinion = cleansing_voting(raw_domain_opi)   # "yes" / "no"
                        domain_opinions[domain] = domain_opinion
                        if domain_opinion == "no":
                            advice_prompt = get_consensus_opinion_prompt(domain, syn_report)
                            advice_output, usage = handler.get_output_multiagent(user_input=advice_prompt, temperature=0, system_role=voter)
                            total_prompt_tokens += usage.prompt_tokens
                            total_completion_tokens += usage.completion_tokens
                            revision_advice[domain] = advice_output
                            hasno_flag = True
                    if hasno_flag:
                        revision_prompt = get_revision_prompt(syn_report, revision_advice)
                        revised_analysis, usage = handler.get_output_multiagent(user_input=revision_prompt, temperature=0, system_role="")
                        total_prompt_tokens += usage.prompt_tokens
                        total_completion_tokens += usage.completion_tokens
                        syn_report = cleansing_syn_report(question, options, revised_analysis)
                        revision_history.append(revision_advice)
                        syn_repo_history.append(syn_report)
                    vote_history.append(domain_opinions)
                
                # final answer derivation
                answer_prompt = get_final_answer_prompt_wsyn(syn_report)
                output, usage = handler.get_output_multiagent(user_input=answer_prompt, temperature=0, system_role="")
                total_prompt_tokens += usage.prompt_tokens
                total_completion_tokens += usage.completion_tokens
                ans, output = cleansing_final_output(output)

    end_time = time.time()
    total_time = end_time - start_time

    data_info = {
        'id':qid,
        'realidx': realqid,
        'question': question,
        'options': options,
        'predicted_answer': ans,
        'answer_idx': gold_answer,
        'token_usage': {
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
        },
        'time_elapsed': total_time
    }
    
    return data_info
