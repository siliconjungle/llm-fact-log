import OpenAI from 'openai'
import similarity from 'compute-cosine-similarity'

const API_KEY = 'YOUR_API_KEY'

const openai = new OpenAI({
  apiKey: API_KEY,
})

const parseFactsObj = (factsObj) => {
  if (typeof factsObj !== 'object') {
    throw new Error('facts must be an object')
  }

  if (!Array.isArray(factsObj.facts)) {
    throw new Error('facts.facts must be an array')
  }

  const facts = factsObj.facts

  if (!Array.isArray(facts)) {
    throw new Error('facts must be an array')
  }

  for (let i = 0; i < facts.length; i++) {
    const fact = facts[i]

    if (typeof fact !== 'object') {
      throw new Error('each fact must be an object')
    }

    if (fact.a === undefined) {
      throw new Error('each fact must have a property a')
    }

    if (typeof fact.a !== 'string') {
      throw new Error('each fact.a must be a string')
    }

    if (fact.relationship === undefined) {
      throw new Error('each fact must have a property relationship')
    }

    if (typeof fact.relationship !== 'string') {
      throw new Error('each fact.relationship must be a string')
    }

    if (fact.b === undefined) {
      throw new Error('each fact must have a property b')
    }

    if (typeof fact.b !== 'string') {
      throw new Error('each fact.b must be a string')
    }
  }

  // loop through and turn it all to lowercase
  for (let i = 0; i < facts.length; i++) {
    const fact = facts[i]

    fact.a = fact.a.toLowerCase()
    fact.relationship = fact.relationship.toLowerCase()
    fact.b = fact.b.toLowerCase()
  }

  return facts
}

export const createFacts = async (statement) => {
  try {
    const completion = await openai.chat.completions.create({
      messages: [
        {
          role: 'system',
          content: `You MUST output a JSON list of facts in the following format. { facts: [{ a: string, relationship: string, b: string }] }. Use nouns as the subject and object of the relationship. Use short, formal field naming. For example, if the statement is "John is married to Mary", then the fact would be { a: "john", relationship: "married", b: "mary" }. Replace self reference with "user".`,
        },
        { role: 'user', content: statement },
      ],
      model: 'gpt-4-1106-preview',
      response_format: { type: 'json_object' },
    })

    const result = completion.choices[0].message.content

    const factsObj = JSON.parse(result)

    return parseFactsObj(factsObj)
  } catch (e) {
    console.error(e)
  }

  return []
}

export const createAssertions = async (question) => {
  try {
    const completion = await openai.chat.completions.create({
      messages: [
        {
          role: 'system',
          content: `You MUST output a JSON list of facts in the following format. { facts: [{ a: string, relationship: string, b: string }] }. Use nouns as the subject and object of the relationship. Use short formal field naming. For example, if the question is "Am I married to Mary?", then the fact would be { a: "user", relationship: "married", b: "mary" }. Replace self reference with "user"`,
        },
        { role: 'user', content: question },
      ],
      model: 'gpt-4-1106-preview',
      response_format: { type: 'json_object' },
    })

    const result = completion.choices[0].message.content

    const factsObj = JSON.parse(result)

    return parseFactsObj(factsObj)
  } catch (e) {
    console.error(e)
  }

  return []
}

export const createEmbedding = async (input) => {
  try {
    const response = await openai.embeddings.create({
      model: 'text-embedding-ada-002',
      input,
    })

    return response.data[0].embedding
  } catch (e) {
    console.log('_ERROR_', e)
  }

  return null
}

// The embedding creation is useful for when you want to search for things.
export const addFact = async (facts, embeddings, entities, relationships, fact) => {
  // check if a, relationship and b are in the embeddings.
  // if any of them aren't, add them first.
  if (embeddings[fact.a] === undefined) {
    embeddings[fact.a] = await createEmbedding(fact.a)
  }

  if (entities[fact.a] === undefined) {
    entities[fact.a] = true
  }

  if (embeddings[fact.relationship] === undefined) {
    embeddings[fact.relationship] = await createEmbedding(fact.relationship)
  }

  if (relationships[fact.relationship] === undefined) {
    relationships[fact.relationship] = true
  }

  if (embeddings[fact.b] === undefined) {
    embeddings[fact.b] = await createEmbedding(fact.b)
  }

  if (entities[fact.b] === undefined) {
    entities[fact.b] = true
  }

  facts.push(fact)
}

export const assert = (facts, fact) => {
  for (let i = 0; i < facts.length; i++) {
    const currentFact = facts[i]
    if (currentFact.a === fact.a && currentFact.relationship === fact.relationship && currentFact.b === fact.b) {
      return true
    }
  }

  return false
}

export const findClosestEmbedding = (embeddings, embedding) => {
  let closestDistance = Number.MAX_SAFE_INTEGER
  let closestEmbeddingKey = -1

  for (let i = 0; i < embeddings.length; i++) {
    const currentEmbedding = embeddings[i]
    const distance = similarity(currentEmbedding, embedding)

    if (distance < closestDistance) {
      closestDistance = distance
      closestEmbeddingKey = i
    }
  }

  return closestEmbeddingKey
}

export const transformFact = async (embeddings, entities, relationships, fact) => {
  const aEmbedding = embeddings[fact.a]
  const relationshipEmbedding = embeddings[fact.relationship]
  const bEmbedding = embeddings[fact.b]

  // if all of these exist then just return.
  if (aEmbedding !== undefined && relationshipEmbedding !== undefined && bEmbedding !== undefined) {
    return fact
  }

  // since some of them don't exist. We need to find the closest one on each of the parameters.
  if (aEmbedding === undefined) {
    // get the embeddings for all of the entities.
    const entityEmbeddingKeys = Object.keys(entities)
    const entityEmbeddings = entityEmbeddingKeys.map((entity) => {
      return embeddings[entity]
    })

    // need to generate the embedding for a.
    const currentAEmbedding = await createEmbedding(fact.a)

    const closestAEmbedding = findClosestEmbedding(entityEmbeddings, currentAEmbedding)

    if (closestAEmbedding !== -1) {
      fact.a = entityEmbeddingKeys[closestAEmbedding]
    }
  }

  if (relationshipEmbedding === undefined) {
    // get the embeddings for all of the relationships.
    const relationshipEmbeddingKeys = Object.keys(relationships)

    const relationshipEmbeddings = relationshipEmbeddingKeys.map((relationship) => {
      return embeddings[relationship]
    })

    // need to generate the embedding for relationship.
    const currentRelationshipEmbedding = await createEmbedding(fact.relationship)

    const closestRelationshipEmbedding = findClosestEmbedding(relationshipEmbeddings, currentRelationshipEmbedding)

    if (closestRelationshipEmbedding !== -1) {
      fact.relationship = relationshipEmbeddingKeys[closestRelationshipEmbedding]
    }
  }

  if (bEmbedding === undefined) {
    // get the embeddings for all of the entities.
    const entityEmbeddingKeys = Object.keys(entities)
    const entityEmbeddings = entityEmbeddingKeys.map((entity) => {
      return embeddings[entity]
    })

    // need to generate the embedding for b.
    const currentBEmbedding = await createEmbedding(fact.b)

    const closestBEmbedding = findClosestEmbedding(entityEmbeddings, currentBEmbedding)

    if (closestBEmbedding !== -1) {
      fact.b = entityEmbeddingKeys[closestBEmbedding]
    }
  }

  return fact
}

// at the end of the day you need to search by defined things
// but you could be like - let me search for entities?
// let me search for relationships, etc.

// lets create a bunch of facts.
// then lets create a bunch of assertions and ask.
// the biggest question is how to determine if two questions are the same?
const run = async () => {
  const facts = []
  const embeddings = {}
  const entities = {}
  const relationships = {}

  // your statements here
  const statements = []

  for (let i = 0; i < statements.length; i++) {
    const statement = statements[i]
    const currentFacts = await createFacts(statement)

    for (let j = 0; j < currentFacts.length; j++) {
      const fact = currentFacts[j]
      await addFact(facts, embeddings, entities, relationships, fact)
    }
  }

  console.log(facts)

  // ok then when a list of questions are asked, you should take the following steps.
  // 1. create a list of facts from the questions
  // 2. convert each of the a, relationship and b into embeddings if they don't exist
  // 3. convert a, relationship and b into the closest points in the embedding space.
  // your queries here.
  const queries = []

  console.log('_EMBEDDING_KEYS_', Object.keys(embeddings))

  for (let i = 0; i < queries.length; i++) {
    const query = queries[i]
    const currentFacts = await createAssertions(query)

    console.log('_CURRENT_FACTS_', currentFacts)

    for (let j = 0; j < currentFacts.length; j++) {
      const fact = JSON.parse(JSON.stringify(currentFacts[j]))

      const transformedFact = await transformFact(embeddings, entities, relationships, fact)

      const result = assert(facts, transformedFact)

      console.log(transformedFact, result)
    }
  }
}

run()
