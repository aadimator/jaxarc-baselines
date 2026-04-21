---
title: "Author and Submission Instructions"
source: "https://neurips.cc/Conferences/2014/PaperInformation/AuthorSubmissionInstructions"
published: 2014-05-22
---
This document contains information about the process of submitting a paper to NIPS 2014. You can also find the Call for Papers, evaluation criteria for NIPS papers, a page with the style files, and a complete list of keywords on the NIPS website

## 1\. Paper Submission and Deadline

All submissions must be in PDF format. As we do not accept paper submissions, particular care should be taken to ensure that your paper prints well; please consult Section 6 in the PDF example file. Note that the maximum file size for submissions is 10MB.

You can upload an early version of your paper well before the deadline. If you start registering your paper only a few minutes before the deadline, you may not have enough time to fill in all the forms. Replacing an earlier version later is allowed, and does not take long, so "upload early and often."

The NIPS evaluation criteria form the basis for acceptance decisions. If your paper is accepted, there will be an opportunity to revise accepted manuscripts before the meeting. There will also be a brief period after the meeting in which small changes such as additional references are still allowed. If you are a reviewer as well as an author, please use the same email address to login to the CMT system. If your paper is registered by your co-author, tell them which of your email addresses to use (the same as the one you are using as a reviewer).

CMT FAQ: you can access FAQ information inside the CMT site by typing FAQ into the query box near the top of the page in the author console and then clicking on the FAQs link.

## 2\. Title Format

This section mainly applies to the title field in CMT where you upload your paper. This does not apply to your e.g., LaTeX source code, where unicode may not be acceptable. But the title of the PDF and the metadata title must ultimately match each other exactly.

These are some title guidelines to help your paper be easily discoverable via search engines. Following these should improve your paper's probability of being cited.

1. Technically, you may use any character encodable by unicode in the title.
2. But try to avoid usage of non-western characters. That is, try to only use ascii characters plus for example, üöé. Most search utilities should be able to present results for "Schölkopf" even if you enter "Scholkopf". However, if you search for "theta" some search engines will not match against "θ".
3. If you must use symbols in your title then you must use unicode characters - do not use latex in the metadata title.
	#### Examples of Unicode in Titles (unicode in bold)
	- **Σ** -Optimality for Active Learning on Gaussian Random Fields
		- On the **χ²** Distribution
		- Nystr **ö** m Method vs Random Fourier Features: A Theoretical and Empirical Comparison - note: ASCII equivalent not needed
	#### Better, ASCII Versions
	- Sigma-Optimality for Active Learning on Gaussian Random Fields
		- On the Chi-Squared Distribution
		- Nystr **ö** m Method vs Random Fourier Features: A Theoretical and Empirical Comparison - note: ASCII equivalent not needed

### Title Capitalization

Please use proper title casing. For more information see [http://blog.apastyle.org/apastyle/2012/03/title-case-and-sentence-case-capitalization-in-apa-style.html](http://blog.apastyle.org/apastyle/2012/03/title-case-and-sentence-case-capitalization-in-apa-style.html)

#### Correct Title Capitalization Example

Transfer Learning in a Transductive Setting

#### Incorrect Title Capitalizaiton (anti-example: do not do this):

transfer learning in a transductive setting

## 2\. Abstract Format

The abstract is submitted as plain unicode text, with a maximum of 2000 characters. For abstracts containing mathematical expressions we encourage the use of mathjax. Basically, if the latex math expression is

```
[expression]
```
then the abstract should contain
```
$[expression]$
```
An example would be
```
$O(\sqrt{T})$
```

**It is important to note that, for security reasons, a space is required before the '>' character, and a space is required after the '<' character, should your formulas require those characters.** For more information on mathjax please see [http://www.mathjax.org/](http://www.mathjax.org/).

If unicode can be used instead of LaTeX, unicode should be used instead.

## 3\. Author Format

For all authors, enter the full and correct citation name into CMT. If your co-author uses a middle initial for their own citations, please enter it.

## 4\. Paper Format

Your submission must follow the style files that are available at this location. Papers are limited to eight pages, including figures and tables, in the NIPS style. However, an additional ninth page containing only cited references is allowed. Papers departing from the formatting guidelines, and all papers longer than nine (9) pages, or where the ninth page contains text other than references, will be rejected without review.

The style files are nips14submite.sty, nips14submit09.sty, nips2014.rtf and nips2014.docx.

## 5\. Double Blind Submission

The NIPS reviewing process is double blind; please follow the style files provided and make sure the submission does not include author names and affiliations. The following are some guidelines for submitting double blind papers:

Do not include any author name or affiliation in the title.

Do not include any acknowledgments.

Refer to your own published work in the third person. That is, use "In the previous work of Jones et al. \[2\]", not "In our previous work \[2\]".

If you cite your other papers that are not widely available (e.g., a difficult-to-find technical report or a paper currently under review), use anonymous author names in the citation, e.g., an author of the form "A. Anonymous".

Papers not satisfying these guidelines will not be reviewed.

## 6\. Keywords

The purpose of the keywords is to assist assignment of papers to area chairs and reviewers. However, the separation between different areas does not play a major role, and authors should refrain from choosing keywords based on past rumors of what may increase their chances of acceptance. The keywords are not hierarchically structured so you should simply choose those keywords that best describe your paper. The list of keywords can be found on the NIPS website.

## 7\. Submitting supplementary material

The deadline for supplementary material is the same as for papers, as given in Section 1 above.

In addition to the submitted PDF paper, authors can additionally submit supplementary material for their paper. Please make sure that the supporting material is also anonymized, removing author names from any extra material, and referring to previous work in the third person.

Such extra material may include long technical proofs that do not fit into the paper, image, audio or video sample outputs from your algorithm, animations that describe your algorithm, details of experimental results, or even source code for running experiments.

Note that the reviewers and the program committee reserve the right to judge the paper solely on the basis of the 8 pages, 9 pages including citations, of the paper; looking at any extra material is up to the discretion of the reviewers and is not required.

To submit the supplementary material you should create an archive file. You will be given an opportunity to upload it to CMT after submitting the paper itself. This archive file can be in the.zip format (using WinZip or equivalent), or in.tar.gz or.tgz format (using tar and gzip). Alternatively, you can choose to upload a single PDF file as supplementary material. There is a strict upper limit of 10MB for the supplementary material. Archives larger than this will be rejected by CMT. Please be circumspect, and upload only files that are directly useful for evaluating your paper.

To reduce load on the server, we request that large files not be uploaded right at the time of the deadline. Please upload material that is ready early (e.g., videos or images) as soon as possible.

For your archive, for each extra file, use a file name that starts with a description of the extra material, such as "proof", or "appendix", or "video", and ends with the appropriate extension.

Again, please ensure that the supporting material (including filenames) is anonymized.

## 8\. Policy on dual submission

Submissions that are identical (or substantially similar) to versions that have been previously published, or accepted for publication, or that have been submitted in parallel to other conferences are not appropriate for NIPS and violate our dual submission policy. Exceptions to this rule are the following:

1. Submission is permitted of a short version of a paper that has been submitted to a journal, but has not yet been published in that journal. Authors must declare such dual-submissions either through the CMT submission form, or via email to the program chairs at program-chairs@nips.cc. It is the authors' responsibility to make sure that the journal in question allows dual concurrent submissions to conferences.
2. Submission is permitted for papers presented or to be presented at conferences or workshops without proceedings, or with only abstracts published.
3. It is acceptable to submit to NIPS 2014 work that has been made available as a technical report (or similar, e.g. in arXiv) without citing it.

None of the above should be construed as overriding the requirements of other publishing venues. In addition, keep in mind that author anonymity to NIPS reviewers might be compromised for authors availing themselves of exceptions 2 and 3.

## 9\. Policy on Near-Duplicate Submissions

Multiple submissions with an excessive amount of overlap in their text or technical content are not acceptable. The Program Chairs reserve the right to immediately reject all NIPS submissions which they deem to be excessively similar and by the same authors. Such “shotgun submissions” are unacceptable, unfair to authors who submit single original papers, and place an additional strain on the review process.

## 10\. Author feedback

Starting the first week of August, authors will be given the chance to see the reviews and respond to their content.

Your feedback can be used by the Program Committee to alter their judgment about the paper, so it is in your advantage to respond. The maximum length of the response is 6000 characters.

Authors can submit one response per submission. Once author feedback is enabled, authors will see a link to "Author Feedback" from the author console. Any of the authors of a paper can enter/edit the rebuttal, and the rebuttal can be returned to and edited up to the deadline for author feedback.

Remember that reviewing is double-blind. Do not include any information in the response that can identify you or your co-authors. Please do not include any URLs in your response.

Finally, we recommend using the response judiciously. If a reviewer has expressed uncertainty about an issue, or is making an incorrect assumption, or has misunderstood a point in the paper, that is ideal to address in an author's response. There is no need to respond to every minor question or suggestion for improvement. In any event, the best strategy is to be polite and professional.

## 11\. Paper publication

A near-final version of each accepted paper will be due in advance of the meeting to be included in the electronic conference proceedings. Please note that this year we will publish online the anonymous reviews of all accepted papers. (Reviewers will have the chance to edit these before publication.) As was done last year, authors will have the option to include minor revisions and additional references after the conference, so that they can benefit from feedback on their work during the meeting.

Corinna Cortes and Neil Lawrence  
NIPS 2014 Program Co-chairs  
[program-chairs@nips.cc](mailto:program-chairs@nips.cc)