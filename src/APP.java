import java.io.File;
import java.nio.file.Paths;

import org.apache.lucene.analysis.en.EnglishAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.MultiDocValues;
import org.apache.lucene.index.NumericDocValues;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.Explanation;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.similarities.BM25Similarity;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.SmallFloat;
import org.apache.lucene.queryparser.classic.QueryParser;

import java.nio.file.Paths;
public class APP {
    public static void main(String[] args) throws Exception {
        String indexPath = "/part/01/Tmp/yuchen/indexes/clueweb22b_ikat23_fengran_sparse_index_2/";
        String field = "contents";
        String queryString = "What Turkish souvenirs would you recommend, considering my mother's interest in antique crystals and porcelains?";

        // open the index
        DirectoryReader reader = DirectoryReader.open(FSDirectory.open(Paths.get(indexPath)));
        IndexSearcher searcher = new IndexSearcher(reader);
        searcher.setSimilarity(new BM25Similarity(0.9f, 0.4f)); 

        //  query processing
        EnglishAnalyzer analyzer = new EnglishAnalyzer();
        QueryParser parser = new QueryParser(field, analyzer);
        Query query = parser.parse(queryString);

        System.out.println("Query: " + query.toString(field));

        // search top 10
        TopDocs topDocs = searcher.search(query, 10);
        System.out.println("Top results:");
        for (ScoreDoc sd : topDocs.scoreDocs) {
            Document doc = searcher.doc(sd.doc);
            String docid = doc.get("id");  // depending on your index schema
            System.out.println(docid + " -> " + sd.score);
        }

        // specify the doc internal ID to explain
        int luceneDocId = findLuceneDocId(searcher, "clueweb22-en0005-23-09496:0");

        // Explain BM25 scoring
        Explanation explanation = searcher.explain(query, luceneDocId);
        System.out.println("\nExplanation for docID clueweb22-en0005-23-09496:0:");
        System.out.println(explanation.toString());
        // —— NEW: fetch and decode the stored norm byte to get actual token count —— 
        NumericDocValues norms = MultiDocValues.getNormValues(reader, field);
        if (norms != null && norms.advance(luceneDocId) == luceneDocId) {
            byte normByte = (byte) norms.longValue();
            float f = SmallFloat.byte315ToFloat(normByte);
            float docLength = 1.0f / (f * f);
            System.out.println("Document length (tokens) = " + docLength);
        } else {
            System.out.println("No norms available for field: " + field);
        }

        // Explain BM25 scoring
        Explanation expl = searcher.explain(query, luceneDocId);
        System.out.println("\nExplanation for :");
        System.out.println(expl.toString());

        reader.close();

    }

    // auxiliary function: find internal Lucene docID
    private static int findLuceneDocId(IndexSearcher searcher, String externalDocId) throws Exception {
        Query idQuery = new TermQuery(new Term("id", externalDocId));
        TopDocs hits = searcher.search(idQuery, 1);
        if (hits.totalHits.value == 0) {
            throw new RuntimeException("Doc ID not found: " + externalDocId);
        }
        return hits.scoreDocs[0].doc;
    }
}
